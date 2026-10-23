"""Pair-manifest schema for auditable VLA × world-model composition."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from evaluation.contracts import ActionChunk, ObservationBundle, STATE_SOURCES


VIEW_MODES = frozenset({"single", "mapping", "list", "stack"})


@dataclass(frozen=True)
class VideoSpec:
    """Native output window; inclusive windows share a generated endpoint."""

    fps: float
    frames_per_chunk: int
    temporal_layout: str = "future"

    def __post_init__(self):
        if isinstance(self.fps, bool) or not isinstance(self.fps, (int, float)) or not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Video fps must be finite and positive")
        if type(self.frames_per_chunk) is not int or self.frames_per_chunk < 1:
            raise ValueError("Video frames_per_chunk must be a positive integer")
        if self.temporal_layout not in {"future", "inclusive"} or (self.temporal_layout == "inclusive" and self.frames_per_chunk < 2):
            raise ValueError("Video temporal_layout requires future or an inclusive window of at least two frames")

    @property
    def advance_frames(self):
        return self.frames_per_chunk - (self.temporal_layout == "inclusive")

    def frame_count(self, chunks):
        return 1 + chunks * self.advance_frames

    def assemble(self, initial, chunks):
        """Follow upstream inclusive stitching: first F-1, ..., last F.

        No real initial frame is inserted for inclusive predicted windows.
        The terminal of each raw window remains the recurrent model input,
        even though the next window's prediction owns that saved boundary.
        """
        import numpy as np
        chunks = list(chunks)
        if not chunks:
            return initial[-1:]
        if self.temporal_layout == "inclusive":
            return np.concatenate([value[:-1] for value in chunks[:-1]] + [chunks[-1]])
        return np.concatenate([initial[-1:]] + chunks)

    @classmethod
    def from_dict(cls, value):
        if "final_chunk_valid_frames" in value:
            from evaluation.timeline import TailVideoSpec
            if not {"fps", "frames_per_chunk", "final_chunk_valid_frames"} <= set(value) <= {
                    "fps", "frames_per_chunk", "temporal_layout", "final_chunk_valid_frames"}:
                raise ValueError("Tail video spec contains unknown/missing fields")
            return TailVideoSpec(**value)
        if not {"fps", "frames_per_chunk"} <= set(value) <= {"fps", "frames_per_chunk", "temporal_layout"}:
            raise ValueError("Video spec requires exactly fps and frames_per_chunk")
        return cls(**value)

    def validate_output(self, output, names):
        if set(output.views) != set(names):
            raise ValueError("Generated video views do not match declared output_views")
        for name, frames in output.views.items():
            if len(frames) != self.frames_per_chunk:
                raise ValueError(f"View {name!r} returned {len(frames)} frames; expected {self.frames_per_chunk} in {self.temporal_layout} layout")


def _required(mapping: Mapping[str, Any], key: str, *, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{context} is missing required field {key!r}")
    return mapping[key]


@dataclass(frozen=True)
class ActionSpec:
    dimension: int
    horizon: int
    representation: str
    frame: str
    semantics: str
    rate_hz: float
    normalization: str = "none"
    components: tuple[str, ...] = ()
    temporal_layout: str = "future"

    def __post_init__(self) -> None:
        if self.dimension < 1 or self.horizon < 1:
            raise ValueError("Action dimension and horizon must be positive")
        if self.rate_hz <= 0:
            raise ValueError("Action rate_hz must be positive")
        for name in ("representation", "frame", "semantics", "normalization"):
            if not getattr(self, name):
                raise ValueError(f"ActionSpec.{name} must be non-empty")
        if self.components and len(self.components) != self.dimension:
            raise ValueError(
                f"ActionSpec.components has {len(self.components)} names for "
                f"dimension {self.dimension}"
            )
        object.__setattr__(self, "components", tuple(self.components))
        if self.temporal_layout not in {"future", "inclusive"} or (self.temporal_layout == "inclusive" and self.horizon < 2):
            raise ValueError("Action temporal_layout requires future or an inclusive window of at least two samples")

    @property
    def advance_steps(self):
        return self.horizon - (self.temporal_layout == "inclusive")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionSpec":
        return cls(
            dimension=int(_required(value, "dimension", context="action spec")),
            horizon=int(_required(value, "horizon", context="action spec")),
            representation=str(_required(value, "representation", context="action spec")),
            frame=str(_required(value, "frame", context="action spec")),
            semantics=str(_required(value, "semantics", context="action spec")),
            rate_hz=float(_required(value, "rate_hz", context="action spec")),
            normalization=str(value.get("normalization", "none")),
            components=tuple(value.get("components", ())),
            temporal_layout=str(value.get("temporal_layout", "future")),
        )

    def wrap(self, value: Any, *, source: str) -> ActionChunk:
        action = ActionChunk.from_value(
            value,
            representation=self.representation,
            frame=self.frame,
            semantics=self.semantics,
            source=source,
            rate_hz=self.rate_hz,
            component_names=self.components,
            normalization=self.normalization,
            temporal_layout=self.temporal_layout,
        )
        if action.dimension != self.dimension:
            raise ValueError(
                f"{source} action width is {action.dimension}, expected {self.dimension}"
            )
        if action.horizon != self.horizon:
            raise ValueError(
                f"{source} action horizon is {action.horizon}, expected {self.horizon}"
            )
        if action.temporal_layout != self.temporal_layout:
            raise ValueError("Action chunk temporal layout differs from its declared protocol")
        return action


@dataclass(frozen=True)
class AdapterSpec:
    adapter: str
    view_mapping: Mapping[str, str]
    view_mode: str
    squeeze_single_frame: bool
    action: ActionSpec
    state_schema: str | None = None
    output_views: tuple[str, ...] = ()
    image_size: tuple[int, int] | None = None
    video: VideoSpec | None = None
    history_offsets: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.adapter:
            raise ValueError("AdapterSpec.adapter must be non-empty")
        if self.view_mode not in VIEW_MODES:
            raise ValueError(f"Unsupported view_mode {self.view_mode!r}")
        mapping = {str(key): str(value) for key, value in self.view_mapping.items()}
        if not mapping or any(not key or not value for key, value in mapping.items()):
            raise ValueError("AdapterSpec.view_mapping must contain non-empty names")
        if self.view_mode == "single" and len(mapping) != 1:
            raise ValueError("single view_mode requires exactly one view mapping")
        object.__setattr__(self, "view_mapping", mapping)
        object.__setattr__(self, "output_views", tuple(self.output_views))
        if self.history_offsets is not None:
            offsets = tuple(self.history_offsets)
            if (not offsets or any(type(x) is not int or x > 0 for x in offsets)
                    or tuple(sorted(set(offsets))) != offsets or offsets[-1] != 0):
                raise ValueError("history_offsets must be increasing unique nonpositive integers ending at 0")
            object.__setattr__(self, "history_offsets", offsets)
        if self.image_size is not None:
            size = tuple(int(x) for x in self.image_size)
            if len(size) != 2 or min(size) < 1:
                raise ValueError("image_size must be positive [width, height]")
            object.__setattr__(self, "image_size", size)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, context: str) -> "AdapterSpec":
        return cls(
            adapter=str(_required(value, "adapter", context=context)),
            view_mapping=dict(_required(value, "view_mapping", context=context)),
            view_mode=str(value.get("view_mode", "mapping")),
            squeeze_single_frame=bool(value.get("squeeze_single_frame", False)),
            action=ActionSpec.from_dict(_required(value, "action", context=context)),
            state_schema=value.get("state_schema"),
            output_views=tuple(value.get("output_views", ())),
            image_size=value.get("image_size"),
            video=VideoSpec.from_dict(value["video"]) if value.get("video") is not None else None,
            history_offsets=value.get("history_offsets"),
        )

    @property
    def required_bundle_views(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.view_mapping.values()))

    def adapter_input(self, observation: ObservationBundle) -> Any:
        if self.history_offsets is not None:
            import numpy as np
            views = {}
            for name in self.required_bundle_views:
                history = observation.views[name]
                indices = np.asarray(self.history_offsets) + len(history) - 1
                if np.any(indices < 0):
                    raise ValueError(f"View {name!r} lacks required history {self.history_offsets}")
                views[name] = history[indices]
            from dataclasses import replace
            observation = replace(observation, views=views)
        if self.image_size is not None:
            import numpy as np
            from PIL import Image

            observation = ObservationBundle(
                views={name: np.stack([
                    np.asarray(Image.fromarray(frame).resize(self.image_size, Image.Resampling.BILINEAR))
                    for frame in history
                ]) for name, history in observation.views.items() if name in self.required_bundle_views},
                instruction=observation.instruction,
                robot_state=observation.robot_state,
                step_index=observation.step_index,
                timestamp=observation.timestamp,
                metadata=observation.metadata,
            )
        return observation.adapter_views(
            self.view_mapping,
            mode=self.view_mode,
            squeeze_single_frame=self.squeeze_single_frame,
        )


@dataclass(frozen=True)
class ActionBridgeSpec:
    name: str
    source: ActionSpec
    target: ActionSpec
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ActionBridgeSpec.name must be non-empty")
        object.__setattr__(self, "config", dict(self.config))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionBridgeSpec":
        return cls(
            name=str(_required(value, "name", context="action_bridge")),
            source=ActionSpec.from_dict(_required(value, "source", context="action_bridge")),
            target=ActionSpec.from_dict(_required(value, "target", context="action_bridge")),
            config=dict(value.get("config", {})),
        )


@dataclass(frozen=True)
class StateProviderSpec:
    type: str
    state_schema: str
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in STATE_SOURCES:
            raise ValueError(
                f"Unknown state provider {self.type!r}; expected one of {sorted(STATE_SOURCES)}"
            )
        if not self.state_schema:
            raise ValueError("StateProviderSpec.state_schema must be non-empty")
        object.__setattr__(self, "config", dict(self.config))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StateProviderSpec":
        return cls(
            type=str(_required(value, "type", context="state_provider")),
            state_schema=str(_required(value, "state_schema", context="state_provider")),
            config=dict(value.get("config", {})),
        )


@dataclass(frozen=True)
class ViewBridgeSpec:
    name: str = "identity"
    config: Mapping[str, Any] = field(default_factory=dict)
    output_views: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ViewBridgeSpec.name must be non-empty")
        object.__setattr__(self, "config", dict(self.config))
        names = tuple(self.output_views)
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("ViewBridge output_views must be unique nonempty names")
        object.__setattr__(self, "output_views", names)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "ViewBridgeSpec":
        value = value or {}
        return cls(name=str(value.get("name", "identity")), config=dict(value.get("config", {})),
                   output_views=tuple(value.get("output_views", ())))


@dataclass(frozen=True)
class PairManifest:
    pair_id: str
    vla: AdapterSpec
    world_model: AdapterSpec
    action_bridge: ActionBridgeSpec
    state_provider: StateProviderSpec
    view_bridge: ViewBridgeSpec = field(default_factory=ViewBridgeSpec)
    schema_version: int = 1
    seed: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def video_output_views(self):
        """Views after observation conversion, not only the raw WM canvas."""
        return self.view_bridge.output_views or self.world_model.output_views

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported pair-manifest schema_version {self.schema_version}")
        if not self.pair_id:
            raise ValueError("PairManifest.pair_id must be non-empty")
        if self.vla.action != self.action_bridge.source:
            raise ValueError("VLA action spec must exactly match action_bridge.source")
        if self.world_model.action != self.action_bridge.target:
            raise ValueError("World-model action spec must exactly match action_bridge.target")
        if self.world_model.adapter.rsplit(".", 1)[-1] == "CtrlWorldAdapter":
            video = self.world_model.video
            if (self.world_model.action.temporal_layout != "inclusive" or video is None
                    or video.temporal_layout != "inclusive" or video.frames_per_chunk != 5):
                raise ValueError("Legacy Ctrl timing is unsupported: declare inclusive 5-sample action/video windows (12 source ticks)")
        if self.state_provider.type == "world_model_prediction":
            if self.world_model.state_schema != self.state_provider.state_schema:
                raise ValueError(
                    "world_model_prediction requires matching world-model and provider state schemas"
                )
        if self.state_provider.type == "none" and self.vla.state_schema:
            raise ValueError("none state provider cannot serve a state-dependent VLA")
        if self.state_provider.type == "pilworld_action_target":
            from evaluation.modes import pilworld_alignment
            pilworld_alignment(self.vla.action, self.world_model.action)
            if self.vla.state_schema != self.state_provider.state_schema:
                raise ValueError("pilworld requires matching policy and provider state schemas")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PairManifest":
        return cls(
            schema_version=int(value.get("schema_version", 1)),
            pair_id=str(_required(value, "pair_id", context="pair manifest")),
            vla=AdapterSpec.from_dict(
                _required(value, "vla", context="pair manifest"), context="vla"
            ),
            world_model=AdapterSpec.from_dict(
                _required(value, "world_model", context="pair manifest"),
                context="world_model",
            ),
            action_bridge=ActionBridgeSpec.from_dict(
                _required(value, "action_bridge", context="pair manifest")
            ),
            state_provider=StateProviderSpec.from_dict(
                _required(value, "state_provider", context="pair manifest")
            ),
            view_bridge=ViewBridgeSpec.from_dict(value.get("view_bridge")),
            seed=int(value.get("seed", 0)),
            metadata=dict(value.get("metadata", {})),
        )

    def validate_adapters(self, vla: Any, world_model: Any) -> None:
        actual_vla = f"{type(vla).__module__}.{type(vla).__name__}"
        actual_world_model = f"{type(world_model).__module__}.{type(world_model).__name__}"
        if self.vla.adapter not in {type(vla).__name__, actual_vla, getattr(vla, "adapter_identity", None)}:
            raise ValueError(
                f"Pair manifest expects VLA {self.vla.adapter}, got {actual_vla}"
            )
        if self.world_model.adapter not in {type(world_model).__name__, actual_world_model, getattr(world_model, "adapter_identity", None)}:
            raise ValueError(
                f"Pair manifest expects world model {self.world_model.adapter}, "
                f"got {actual_world_model}"
            )

    def validate_observation(self, observation: ObservationBundle) -> None:
        observation.require_views(self.vla.required_bundle_views)
        observation.require_views(self.world_model.required_bundle_views)
        if self.vla.state_schema:
            if observation.robot_state is None:
                raise ValueError(
                    f"VLA {self.vla.adapter} requires state schema {self.vla.state_schema!r}"
                )
            if observation.robot_state.schema != self.vla.state_schema:
                raise ValueError(
                    f"VLA state schema is {observation.robot_state.schema!r}, "
                    f"expected {self.vla.state_schema!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_pair_manifest(path: str | Path) -> PairManifest:
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError("Pair manifests currently use JSON so validation stays dependency-light")
    return PairManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def pair_for_job(pair: PairManifest, job: Mapping[str, Any]) -> PairManifest:
    """Resolve a suite's state feedback without silently changing the protocol."""
    from dataclasses import replace
    from evaluation.modes import POLICY_MODES
    if job["mode"] not in POLICY_MODES:
        return pair
    provider = job["state_provider"]
    if (job["mode"] == "pilworld") != (provider == "pilworld_action_target"):
        raise ValueError("pilworld mode requires pilworld_action_target, and vice versa")
    config = {"action": "native", "alignment": "stride_aligned_terminal"} if job["mode"] == "pilworld" else pair.state_provider.config
    return replace(pair, state_provider=StateProviderSpec(provider, pair.state_provider.state_schema, config),
        metadata={**pair.metadata, "evaluation_mode": job["mode"],
                  "pilworld_paper_reproduction": False})


__all__ = [
    "ActionBridgeSpec",
    "ActionSpec",
    "AdapterSpec",
    "PairManifest",
    "StateProviderSpec",
    "ViewBridgeSpec",
    "load_pair_manifest",
]
