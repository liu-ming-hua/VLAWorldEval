"""Explicit next-state providers for protocol-aware closed-loop evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from evaluation.contracts import ActionChunk, ObservationBundle, RobotState, WorldModelOutput
from evaluation.manifests import StateProviderSpec
from evaluation.modes import pilworld_alignment


@dataclass(frozen=True)
class StateContext:
    step_index: int
    previous_observation: ObservationBundle
    world_model_output: WorldModelOutput
    native_action: ActionChunk
    bridged_action: ActionChunk
    world_model: Any
    trial: Mapping[str, Any]


class StateProvider(ABC):
    """Resolve the state used by the next VLA query."""

    source: str

    def __init__(self, spec: StateProviderSpec):
        self.spec = spec
        self.source = spec.type

    def reset(self, initial_observation: ObservationBundle, trial: Mapping[str, Any]) -> None:
        return None

    @abstractmethod
    def provide(self, context: StateContext) -> RobotState | None:
        pass


def _component_slices(values: np.ndarray, config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    slices = config.get("component_slices")
    if not slices:
        return {str(config.get("state_key", "state")): values.copy()}
    result = {}
    for key, bounds in slices.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"component_slices[{key!r}] must be [start, end]")
        start, end = int(bounds[0]), int(bounds[1])
        if start < 0 or end <= start or end > values.shape[-1]:
            raise ValueError(
                f"Invalid component slice {key!r}={bounds} for vector width {values.shape[-1]}"
            )
        result[str(key)] = values[..., start:end].copy()
    return result


class DatasetGroundTruthProvider(StateProvider):
    """Read the next measured state from ``trial['state_sequence']``."""

    def reset(self, initial_observation: ObservationBundle, trial: Mapping[str, Any]) -> None:
        if "state_sequence" not in trial:
            raise ValueError(
                "dataset_ground_truth requires trial['state_sequence']; "
                "oracle state cannot be inferred from actions"
            )
        self.sequence = trial["state_sequence"]

    def provide(self, context: StateContext) -> RobotState:
        index = context.step_index + 1
        if isinstance(self.sequence, Mapping):
            values = {}
            for key, sequence in self.sequence.items():
                array = np.asarray(sequence)
                if index >= len(array):
                    raise IndexError(f"Ground-truth state {key!r} has no step {index}")
                values[str(key)] = array[index]
        else:
            if index >= len(self.sequence):
                raise IndexError(f"Ground-truth state sequence has no step {index}")
            values = self.sequence[index]
        return RobotState.from_value(
            values,
            schema=self.spec.state_schema,
            source=self.source,
            metadata={"sequence_index": index},
        )


class NoStateProvider(StateProvider):
    """For image/text policies only: absence is not a synthesized robot state."""

    def provide(self, context):
        return None


class HoldInitialProvider(StateProvider):
    """Explicit ablation: retain the initial measured state, never label it GT."""

    def reset(self, initial_observation, trial):
        if initial_observation.robot_state is None:
            raise ValueError("hold_initial requires an initial state")
        self.initial = initial_observation.robot_state

    def provide(self, context):
        return RobotState(values=self.initial.values, schema=self.spec.state_schema,
                          source="hold_initial", metadata={"held_from_step": 0})


class LastActionTargetProvider(StateProvider):
    """Reuse the terminal absolute action target as the next state."""

    def provide(self, context: StateContext) -> RobotState:
        action_name = str(self.spec.config.get("action", "native"))
        action = context.native_action if action_name == "native" else context.bridged_action
        if action.semantics != "absolute_target" and not self.spec.config.get(
            "allow_non_absolute", False
        ):
            raise ValueError(
                "last_action_target is only valid for absolute_target actions; "
                f"received semantics={action.semantics!r}"
            )
        values = _component_slices(action.values[-1], self.spec.config)
        return RobotState(
            values=values,
            schema=self.spec.state_schema,
            source=self.source,
            metadata={"action": action_name, "action_source": action.source},
        )


class PilWorldActionTargetProvider(StateProvider):
    """PiL-World-style absolute endpoint proxy, never WM-predicted state.

    The embodiment bridge must map the actual terminal target to policy state.
    This deliberately cannot fall back to velocity integration or H_pi's tail.
    """

    def __init__(self, spec, target_resolver=None):
        super().__init__(spec)
        if spec.config.get("action", "native") != "native" or spec.config.get("allow_non_absolute", False):
            raise ValueError("pilworld_action_target only supports native absolute targets")
        if spec.config.get("alignment", "stride_aligned_terminal") != "stride_aligned_terminal":
            raise ValueError("pilworld requires stride_aligned_terminal alignment")
        if not callable(target_resolver):
            raise ValueError("pilworld_action_target requires an explicit bridge.pilworld_state mapper")
        self.target_resolver = target_resolver

    def provide(self, context):
        previous = context.previous_observation.robot_state
        if previous is None or previous.schema != self.spec.state_schema:
            raise ValueError("pilworld requires initial policy state with the declared schema")
        alignment = pilworld_alignment(context.native_action, context.bridged_action)
        state = self.target_resolver(previous, context.native_action,
            executed_steps=alignment["executed_action_steps"], state_schema=self.spec.state_schema)
        if (not isinstance(state, RobotState) or state.schema != self.spec.state_schema
                or state.source != self.source):
            raise ValueError("PiL-World mapper must return the declared proxy RobotState schema/source")
        if state.metadata.get("native_action_index") != alignment["native_action_index"]:
            raise ValueError("PiL-World mapper returned a target not aligned to the WM terminal frame")
        return RobotState(values=state.values, schema=state.schema, source=self.source,
            metadata={**state.metadata, **alignment, "step_index": context.step_index,
                      "evaluation_mode": "pilworld", "wm_predicted_state": False,
                      "state_assumption": "terminal_absolute_action_target_proxy"})


class IntegratedActionTargetProvider(StateProvider):
    """Analytically integrate an explicitly declared delta-action vector."""

    def __init__(
        self,
        spec: StateProviderSpec,
        integrator: Callable[[RobotState, ActionChunk, StateContext], Any] | None = None,
    ):
        super().__init__(spec)
        self.integrator = integrator

    def provide(self, context: StateContext) -> RobotState:
        previous = context.previous_observation.robot_state
        if previous is None:
            raise ValueError("integrated_action_target requires a previous robot state")
        action_name = str(self.spec.config.get("action", "native"))
        action = context.native_action if action_name == "native" else context.bridged_action
        if self.integrator is not None:
            values = self.integrator(previous, action, context)
        else:
            if action.semantics != "delta":
                raise ValueError(
                    "The default integrator only accepts semantics='delta'; provide an "
                    "embodiment-specific integrator for pose or absolute actions"
                )
            if tuple(previous.values) != ("state",):
                raise ValueError(
                    "The default integrator only supports one flat 'state' component; "
                    "provide an embodiment-specific integrator"
                )
            state = previous.values["state"].reshape(-1)
            delta = action.values.sum(axis=0).reshape(-1)
            if state.shape != delta.shape:
                raise ValueError(
                    f"Cannot integrate state width {state.size} with action width {delta.size}"
                )
            values = state + delta
        return RobotState.from_value(
            values,
            schema=self.spec.state_schema,
            source=self.source,
            metadata={"action": action_name, "action_source": action.source},
        )


class WorldModelPredictionProvider(StateProvider):
    """Require a native state prediction from the world-model output."""

    def provide(self, context: StateContext) -> RobotState:
        state = context.world_model_output.robot_state
        if state is None:
            raise RuntimeError(
                "state_provider=world_model_prediction, but the world model returned no state"
            )
        if state.schema != self.spec.state_schema:
            raise ValueError(
                f"World-model state schema is {state.schema!r}, expected {self.spec.state_schema!r}"
            )
        if state.source != self.source:
            state = RobotState(
                values=state.values,
                schema=state.schema,
                source=self.source,
                uncertainty=state.uncertainty,
                timestamp=state.timestamp,
                metadata={**state.metadata, "adapter_state_source": state.source},
            )
        return state


class StateCompletionProvider(StateProvider):
    """Run a separately trained StateBridge-style completion model."""

    def __init__(
        self,
        spec: StateProviderSpec,
        predictor: Callable[[StateContext], RobotState | Any],
    ):
        super().__init__(spec)
        self.predictor = predictor

    def provide(self, context: StateContext) -> RobotState:
        prediction = self.predictor(context)
        if isinstance(prediction, RobotState):
            if prediction.schema != self.spec.state_schema:
                raise ValueError(
                    f"State completion schema is {prediction.schema!r}, "
                    f"expected {self.spec.state_schema!r}"
                )
            return RobotState(
                values=prediction.values,
                schema=prediction.schema,
                source=self.source,
                uncertainty=prediction.uncertainty,
                timestamp=prediction.timestamp,
                metadata=prediction.metadata,
            )
        return RobotState.from_value(
            prediction,
            schema=self.spec.state_schema,
            source=self.source,
        )


def build_state_provider(
    spec: StateProviderSpec,
    *,
    integrator: Callable[[RobotState, ActionChunk, StateContext], Any] | None = None,
    predictor: Callable[[StateContext], RobotState | Any] | None = None,
    target_resolver: Callable | None = None,
) -> StateProvider:
    if spec.type == "none":
        return NoStateProvider(spec)
    if spec.type == "dataset_ground_truth":
        return DatasetGroundTruthProvider(spec)
    if spec.type == "hold_initial":
        return HoldInitialProvider(spec)
    if spec.type == "last_action_target":
        return LastActionTargetProvider(spec)
    if spec.type == "pilworld_action_target":
        return PilWorldActionTargetProvider(spec, target_resolver)
    if spec.type == "integrated_action_target":
        return IntegratedActionTargetProvider(spec, integrator=integrator)
    if spec.type == "world_model_prediction":
        return WorldModelPredictionProvider(spec)
    if spec.type == "state_completion_prediction":
        if predictor is None:
            raise ValueError("state_completion_prediction requires a predictor")
        return StateCompletionProvider(spec, predictor)
    raise ValueError(f"Unsupported state provider {spec.type!r}")


__all__ = [
    "DatasetGroundTruthProvider",
    "HoldInitialProvider",
    "NoStateProvider",
    "IntegratedActionTargetProvider",
    "LastActionTargetProvider",
    "PilWorldActionTargetProvider",
    "StateCompletionProvider",
    "StateContext",
    "StateProvider",
    "WorldModelPredictionProvider",
    "build_state_provider",
]
