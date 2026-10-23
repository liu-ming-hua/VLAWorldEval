"""Legacy-compatible and protocol-aware VLA × world-model evaluation loops."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from PIL import Image
from tqdm import tqdm

from adapters.actions.base import ActionBridge, build_action_bridge
from adapters.base import VLAAdapter, WorldModelAdapter
from adapters.observations.base import ViewBridge, build_view_bridge
from evaluation.reporting.artifacts import rollout_artifact_dir, save_rollout_artifacts
from evaluation.contracts import ActionChunk, ObservationBundle, RobotState, WorldModelOutput
from evaluation.manifests import PairManifest, load_pair_manifest
from evaluation.state_providers import StateContext, StateProvider, build_state_provider
from evaluation.datasets.trials import discover_trials


@dataclass
class RolloutTrace:
    """Complete, auditable record of one protocol-aware rollout."""

    manifest: PairManifest
    observations: list[ObservationBundle]
    world_outputs: list[WorldModelOutput] = field(default_factory=list)
    native_actions: list[ActionChunk] = field(default_factory=list)
    bridged_actions: list[ActionChunk] = field(default_factory=list)
    timings: list[dict[str, float]] = field(default_factory=list)

    @property
    def final_state(self) -> RobotState | None:
        return self.observations[-1].robot_state

    def view_videos(self) -> dict[str, np.ndarray]:
        """Concatenate the initial observation and every generated chunk by view."""
        names = set(self.observations[0].views)
        for output in self.world_outputs:
            names &= set(output.views)
        videos = {}
        for name in sorted(names):
            chunks = [self.observations[0].views[name][-1:]]
            chunks.extend(output.views[name] for output in self.world_outputs)
            shapes = {tuple(chunk.shape[1:]) for chunk in chunks}
            if len(shapes) != 1:
                raise ValueError(
                    f"View {name!r} changes spatial shape across rollout: {sorted(shapes)}"
                )
            spec = self.manifest.world_model.video
            videos[name] = spec.assemble(chunks[0], chunks[1:]) if spec else np.concatenate(chunks, axis=0)
        return videos

    def primary_video(self) -> np.ndarray:
        videos = self.view_videos()
        if not videos:
            raise RuntimeError("No view persists from the initial observation through the rollout")
        preferred = self.manifest.world_model.output_views
        name = next((candidate for candidate in preferred if candidate in videos), sorted(videos)[0])
        return videos[name]


class WorldModelReplayEvaluator:
    """WM-native action replay; no VLA calls or fictional recurrent GT state."""

    def __init__(self, world_model, manifest):
        self.world_model, self.manifest = world_model, manifest
        identity = getattr(world_model, "adapter_identity", type(world_model).__name__)
        if identity != manifest.world_model.adapter:
            raise ValueError(f"World-model adapter mismatch: {identity} != {manifest.world_model.adapter}")
        self.view_bridge = build_view_bridge(manifest.view_bridge)

    def rollout_detailed(self, trial, actions, *, on_step=None, action_source="dataset_replay"):
        observation = self.view_bridge.prepare_observation(
            PolicyWorldModelEvaluator._initial_bundle(trial, self.manifest))
        self.world_model.reset(self.manifest.world_model.adapter_input(observation))
        trace = RolloutTrace(self.manifest, [observation])
        for index, value in enumerate(actions):
            action = self.manifest.world_model.action.wrap(value, source=action_source)
            started = time.perf_counter()
            output = self.world_model.step_output(value, output_views=self.manifest.world_model.output_views,
                                                 state_schema=self.manifest.world_model.state_schema)
            if self.manifest.world_model.video is not None:
                self.manifest.world_model.video.validate_output(output, self.manifest.world_model.output_views)
            output = self.view_bridge.process_world_output(output)
            if self.manifest.world_model.video is not None:
                self.manifest.world_model.video.validate_output(output, self.manifest.video_output_views)
            trace.bridged_actions.append(action)
            trace.world_outputs.append(output)
            trace.observations.append(observation.next(views=output.terminal_views, robot_state=None))
            observation = trace.observations[-1]
            trace.timings.append({"world_model_seconds": time.perf_counter() - started, "policy_seconds": 0.0})
            if on_step is not None:
                on_step(trace)
        return trace


class PolicyWorldModelEvaluator:
    """Generic VLA × world-model evaluator.

    Without a pair manifest this class preserves the original single-view loop.
    Supplying a manifest enables strict protocol mode: native and bridged
    actions are distinct, observations are named bundles, and a declared
    ``StateProvider`` is mandatory on every recurrent step.
    """

    def __init__(
        self,
        vla: VLAAdapter,
        world_model: WorldModelAdapter,
        scorer: Callable[[np.ndarray, dict], float] | None = None,
        *,
        pair_manifest: PairManifest | str | Path | None = None,
        action_bridge: ActionBridge | None = None,
        state_provider: StateProvider | None = None,
        view_bridge: ViewBridge | None = None,
    ):
        self.vla, self.world_model, self.scorer = vla, world_model, scorer
        if isinstance(pair_manifest, (str, Path)):
            pair_manifest = load_pair_manifest(pair_manifest)
        self.pair_manifest = pair_manifest
        self.action_bridge = action_bridge
        self.state_provider = state_provider
        self.view_bridge = view_bridge

        if self.pair_manifest is None:
            if action_bridge is not None or state_provider is not None or view_bridge is not None:
                raise ValueError("action_bridge/state_provider/view_bridge require a pair_manifest")
        else:
            self.pair_manifest.validate_adapters(vla, world_model)
            if self.action_bridge is None:
                self.action_bridge = build_action_bridge(self.pair_manifest.action_bridge)
            if self.state_provider is None:
                self.state_provider = build_state_provider(
                    self.pair_manifest.state_provider,
                    integrator=getattr(self.action_bridge, "integrate_state", None),
                    target_resolver=getattr(self.action_bridge, "pilworld_state", None),
                )
            if self.view_bridge is None:
                self.view_bridge = build_view_bridge(self.pair_manifest.view_bridge)
            if self.action_bridge.spec != self.pair_manifest.action_bridge:
                raise ValueError("ActionBridge spec does not match the pair manifest")
            if self.state_provider.spec != self.pair_manifest.state_provider:
                raise ValueError("StateProvider spec does not match the pair manifest")
            if self.view_bridge.spec != self.pair_manifest.view_bridge:
                raise ValueError("ViewBridge spec does not match the pair manifest")

    def rollout(
        self,
        trial: dict,
        rollout_length: int = 40,
        *,
        return_actions: bool = False,
    ) -> tuple[np.ndarray, Any] | tuple[np.ndarray, Any, list[np.ndarray]]:
        """Run a rollout while preserving the original public return type."""
        if self.pair_manifest is not None:
            trace = self.rollout_detailed(trial, rollout_length=rollout_length)
            video = trace.primary_video()
            state = trace.final_state.as_policy_input() if trace.final_state is not None else None
            actions = [chunk.values.copy() for chunk in trace.bridged_actions]
            if return_actions:
                return video, state, actions
            return video, state
        return self._rollout_legacy(
            trial,
            rollout_length=rollout_length,
            return_actions=return_actions,
        )

    def rollout_detailed(self, trial: dict, rollout_length: int = 40, *, on_step=None) -> RolloutTrace:
        """Run strict protocol mode and return all intermediate values."""
        if (
            self.pair_manifest is None
            or self.action_bridge is None
            or self.state_provider is None
            or self.view_bridge is None
        ):
            raise RuntimeError(
                "rollout_detailed requires pair_manifest, action_bridge, view_bridge, "
                "and state_provider"
            )
        if rollout_length < 1:
            raise ValueError(f"rollout_length must be positive, got {rollout_length}")

        manifest = self.pair_manifest
        observation = self.view_bridge.prepare_observation(
            self._initial_bundle(trial, manifest)
        )
        manifest.validate_observation(observation)
        reset_input = trial.get("episode_bundle")
        if reset_input is None:
            reset_input = manifest.world_model.adapter_input(observation)
        self.world_model.reset(reset_input)
        initial_state = (
            observation.robot_state.as_policy_input()
            if observation.robot_state is not None
            else None
        )
        self.vla.reset(initial_state)
        self.state_provider.reset(observation, trial)
        trace = RolloutTrace(manifest=manifest, observations=[observation])

        for step_index in range(rollout_length):
            manifest.validate_observation(observation)
            started = time.perf_counter()
            native_value = self.vla.predict_observation(observation, manifest.vla)
            policy_seconds = time.perf_counter() - started
            native_action = manifest.vla.action.wrap(
                native_value,
                source=f"{type(self.vla).__module__}.{type(self.vla).__name__}",
            )
            bridged_action = self.action_bridge.convert(
                native_action,
                observation=observation,
                step_index=step_index,
            )
            manifest.world_model.action.wrap(
                bridged_action.values,
                source=bridged_action.source,
            )

            started = time.perf_counter()
            world_output = self.world_model.step_output(
                bridged_action.values,
                output_views=manifest.world_model.output_views,
                state_schema=manifest.world_model.state_schema,
            )
            if manifest.world_model.video is not None:
                manifest.world_model.video.validate_output(world_output, manifest.world_model.output_views)
            world_output = self.view_bridge.process_world_output(world_output)
            if manifest.world_model.video is not None:
                manifest.world_model.video.validate_output(world_output, manifest.video_output_views)
            world_model_seconds = time.perf_counter() - started
            next_state = self.state_provider.provide(
                StateContext(
                    step_index=step_index,
                    previous_observation=observation,
                    world_model_output=world_output,
                    native_action=native_action,
                    bridged_action=bridged_action,
                    world_model=self.world_model,
                    trial=trial,
                )
            )
            if next_state is None and manifest.state_provider.type != "none":
                raise ValueError("Required state provider returned no state")
            if next_state is not None and next_state.source != manifest.state_provider.type:
                raise ValueError(
                    f"State provider returned source {next_state.source!r}, expected "
                    f"{manifest.state_provider.type!r}"
                )
            next_views = world_output.terminal_views
            if manifest.vla.history_offsets is not None and min(manifest.vla.history_offsets) < 0:
                depth = 1 - min(manifest.vla.history_offsets)
                inclusive = manifest.world_model.video is not None and manifest.world_model.video.temporal_layout == "inclusive"
                next_views = {name: np.concatenate([observation.views[name], frames[1:] if inclusive else frames])[-depth:]
                              for name, frames in world_output.views.items()}
            observation = observation.next(
                views=next_views,
                robot_state=next_state,
                metadata={"world_model": type(self.world_model).__name__},
            )
            trace.native_actions.append(native_action)
            trace.bridged_actions.append(bridged_action)
            trace.world_outputs.append(world_output)
            trace.observations.append(observation)
            trace.timings.append(
                {
                    "policy_seconds": policy_seconds,
                    "world_model_seconds": world_model_seconds,
                }
            )
            if on_step is not None:
                on_step(trace)
        return trace

    def _rollout_legacy(
        self,
        trial: dict,
        *,
        rollout_length: int,
        return_actions: bool,
    ) -> tuple[np.ndarray, Any] | tuple[np.ndarray, Any, list[np.ndarray]]:
        if rollout_length < 1:
            raise ValueError(f"rollout_length must be positive, got {rollout_length}")

        reset_input = trial.get("episode_bundle")
        image = self._initial_observation(trial, reset_input)
        if reset_input is None:
            reset_input = image
        self.world_model.reset(reset_input)
        state = trial.get("initial_state")
        self.vla.reset(state)
        frames = [image]
        action_chunks = []
        for _ in range(rollout_length):
            action = self.vla.predict(frames[-1], trial["instruction"], state)
            if hasattr(action, "detach"):
                action_record = action.detach().float().cpu().numpy().copy()
            else:
                action_record = np.asarray(action, dtype=np.float32).copy()
            action_chunks.append(action_record)
            image = self._validate_observation(self.world_model.step(action))
            model_state = getattr(self.world_model, "last_policy_state", None)
            if model_state is not None:
                state = np.asarray(model_state, dtype=np.float32).copy()
            else:
                state = self.vla.update_state(state, action, image)
            frames.append(image)
        video = np.stack(frames)
        if return_actions:
            return video, state, action_chunks
        return video, state

    @classmethod
    def _initial_bundle(cls, trial: Mapping[str, Any], manifest: PairManifest) -> ObservationBundle:
        candidate = trial.get("observation_bundle")
        if isinstance(candidate, ObservationBundle):
            return candidate
        if isinstance(candidate, Mapping):
            views = candidate.get("views")
            if not isinstance(views, Mapping):
                raise ValueError("observation_bundle mapping requires a 'views' mapping")
            state = candidate.get("robot_state", trial.get("initial_state"))
            if state is not None and not isinstance(state, RobotState):
                state = RobotState.from_value(
                    state,
                    schema=str(
                        candidate.get(
                            "state_schema",
                            manifest.vla.state_schema or manifest.state_provider.state_schema,
                        )
                    ),
                    source=str(candidate.get("state_source", "dataset_ground_truth")),
                )
            return ObservationBundle(
                views=views,
                instruction=str(candidate.get("instruction", trial["instruction"])),
                robot_state=state,
                timestamp=candidate.get("timestamp"),
                metadata=dict(candidate.get("metadata", {})),
            )

        policy_image = trial.get("policy_image", trial.get("trial_png"))
        if isinstance(policy_image, Mapping):
            views = policy_image
        else:
            required = set(manifest.vla.required_bundle_views) | set(
                manifest.world_model.required_bundle_views
            )
            if len(required) != 1:
                raise ValueError(
                    "A multi-view pair requires trial['observation_bundle'] with named views; "
                    "one policy_image cannot be duplicated implicitly"
                )
            views = {next(iter(required)): cls._load_image(policy_image)}
        state = trial.get("initial_state")
        if state is not None and not isinstance(state, RobotState):
            state = RobotState.from_value(
                state,
                schema=manifest.vla.state_schema or manifest.state_provider.state_schema,
                source=str(trial.get("initial_state_source", "dataset_ground_truth")),
            )
        return ObservationBundle(
            views=views,
            instruction=str(trial["instruction"]),
            robot_state=state,
        )

    @staticmethod
    def _load_image(candidate: Any) -> np.ndarray:
        if candidate is None:
            raise ValueError("No policy image was provided")
        if isinstance(candidate, (str, Path)):
            return np.asarray(Image.open(candidate).convert("RGB").resize((256, 256)))
        return PolicyWorldModelEvaluator._validate_observation(candidate)

    @staticmethod
    def _initial_observation(trial: dict, reset_input: Any) -> np.ndarray:
        """Resolve the image shown to a legacy VLA at rollout start."""
        candidate = trial.get("policy_image", trial.get("trial_png"))
        if candidate is not None:
            return PolicyWorldModelEvaluator._load_image(candidate)
        if reset_input is not None:
            return PolicyWorldModelEvaluator._validate_observation(reset_input)
        raise ValueError("A trial needs trial_png, policy_image, or an image episode_bundle")

    @staticmethod
    def _validate_observation(observation: Any) -> np.ndarray:
        if hasattr(observation, "detach"):
            observation = observation.detach().cpu().numpy()
        image = np.asarray(observation)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                "The policy/world-model observation must be an RGB HWC array; "
                f"got shape {image.shape}"
            )
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image * 255 if image.max(initial=0) <= 1.0 else image, 0, 255)
        return np.ascontiguousarray(image, dtype=np.uint8)

    def evaluate(
        self,
        root_dir: str,
        rollout_length: int = 40,
        retries: int = 1,
        video_out_dir: str | None = None,
    ) -> list[dict]:
        if retries < 1:
            raise ValueError(f"retries must be positive, got {retries}")
        trials = discover_trials(root_dir)
        if not trials:
            raise ValueError(f"No valid trial_png + JSON pairs found under {root_dir}")
        results = []
        try:
            for trial in tqdm(trials, desc="VLA x WorldModel trials"):
                for retry_index in range(retries):
                    trace = None
                    if self.pair_manifest is not None:
                        trace = self.rollout_detailed(trial, rollout_length)
                        video = trace.primary_video()
                        action_chunks = [chunk.values for chunk in trace.bridged_actions]
                    else:
                        video, _, action_chunks = self.rollout(
                            trial,
                            rollout_length,
                            return_actions=True,
                        )
                    artifact_paths = {}
                    if video_out_dir:
                        output_dir = rollout_artifact_dir(
                            video_out_dir,
                            root_dir,
                            trial,
                            retry_index,
                        )
                        artifact_paths = save_rollout_artifacts(
                            output_dir,
                            video,
                            action_chunks,
                            trial,
                            native_action_chunks=None
                            if trace is None
                            else [chunk.values for chunk in trace.native_actions],
                            states=None
                            if trace is None
                            else [observation.robot_state for observation in trace.observations],
                            view_videos=None if trace is None else trace.view_videos(),
                            pair_manifest=None if trace is None else trace.manifest.to_dict(),
                            timings=None if trace is None else trace.timings,
                            action_bridge_metadata=None if trace is None else
                            [chunk.metadata for chunk in trace.bridged_actions],
                        )
                    score = self.scorer(video, trial) if self.scorer else None
                    results.append(
                        {
                            "task_key": trial["task_key"],
                            "task_display": trial["task_display"],
                            "score": score,
                            "retry_index": retry_index,
                            "vla_environment": getattr(self.vla, "environment_key", "unknown"),
                            "world_model_environment": getattr(
                                self.world_model, "environment_key", "unknown"
                            ),
                            "pair_id": None
                            if self.pair_manifest is None
                            else self.pair_manifest.pair_id,
                            **artifact_paths,
                        }
                    )
        finally:
            close = getattr(self.world_model, "close", None)
            if callable(close):
                close()
        return results


__all__ = ["PolicyWorldModelEvaluator", "RolloutTrace"]
