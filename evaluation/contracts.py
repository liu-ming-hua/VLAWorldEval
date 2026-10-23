"""Typed, dependency-light contracts shared by evaluators and adapters."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np


STATE_SOURCES = frozenset(
    {
        "none",
        "dataset_ground_truth",
        "hold_initial",
        "integrated_action_target",
        "last_action_target",
        "pilworld_action_target",
        "world_model_prediction",
        "state_completion_prediction",
    }
)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def as_rgb_history(value: Any, *, name: str) -> np.ndarray:
    """Normalize one RGB view to contiguous uint8 ``[T,H,W,3]``."""
    array = _to_numpy(value)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"View {name!r} must be RGB [H,W,3] or [T,H,W,3], got {array.shape}")
    if array.shape[0] < 1 or array.shape[1] < 1 or array.shape[2] < 1:
        raise ValueError(f"View {name!r} cannot contain an empty dimension: {array.shape}")
    if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)):
        raise TypeError(f"View {name!r} must contain real numeric RGB intensities")
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError(f"View {name!r} contains NaN or infinity")
        maximum = float(np.max(array)) if array.size else 0.0
        if float(np.min(array)) < 0 or maximum > 255:
            raise ValueError(f"View {name!r} must be RGB in [0,1] or [0,255]; undo model normalization first")
        if maximum <= 1.0 + 1e-6:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0)
    elif np.min(array) < 0 or np.max(array) > 255:
        raise ValueError(f"View {name!r} integer RGB must be in [0,255]")
    return np.ascontiguousarray(array, dtype=np.uint8)


def _numeric_array(value: Any, *, name: str) -> np.ndarray:
    array = _to_numpy(value)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got dtype {array.dtype}")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return np.ascontiguousarray(array)


@dataclass(frozen=True)
class RobotState:
    """Named proprioceptive state with schema and run-time provenance."""

    values: Mapping[str, np.ndarray]
    schema: str
    source: str
    uncertainty: Mapping[str, np.ndarray] | None = None
    timestamp: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.schema:
            raise ValueError("RobotState.schema must be non-empty")
        if not self.source:
            raise ValueError("RobotState.source must be non-empty")
        if not self.values:
            raise ValueError("RobotState.values must contain at least one component")
        values = {
            str(key): _numeric_array(value, name=f"state component {key!r}")
            for key, value in self.values.items()
        }
        uncertainty = None
        if self.uncertainty is not None:
            uncertainty = {
                str(key): _numeric_array(value, name=f"state uncertainty {key!r}")
                for key, value in self.uncertainty.items()
            }
            unknown = sorted(set(uncertainty) - set(values))
            if unknown:
                raise ValueError(f"State uncertainty has unknown components: {unknown}")
            for key, value in uncertainty.items():
                if value.shape != values[key].shape:
                    raise ValueError(
                        f"Uncertainty for {key!r} has shape {value.shape}, "
                        f"expected {values[key].shape}"
                    )
                if np.any(value < 0):
                    raise ValueError(f"Uncertainty for {key!r} must be non-negative")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        schema: str,
        source: str,
        uncertainty: Any = None,
        timestamp: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "RobotState":
        if isinstance(value, cls):
            return value
        values = value if isinstance(value, Mapping) else {"state": value}
        if uncertainty is None or isinstance(uncertainty, Mapping):
            uncertainty_values = uncertainty
        else:
            uncertainty_values = {"state": uncertainty}
        return cls(
            values=values,
            schema=schema,
            source=source,
            uncertainty=uncertainty_values,
            timestamp=timestamp,
            metadata=metadata or {},
        )

    def as_policy_input(self) -> Any:
        """Return the mapping expected by named-state VLAs, or one flat value."""
        if tuple(self.values) == ("state",):
            value = self.values["state"]
            return value.item() if value.ndim == 0 else value
        return {key: value.copy() for key, value in self.values.items()}

    def flattened(self) -> tuple[np.ndarray, tuple[str, ...], tuple[int, ...]]:
        names = tuple(self.values)
        sizes = tuple(int(self.values[name].size) for name in names)
        values = np.concatenate([self.values[name].reshape(-1) for name in names])
        return values.astype(np.float32, copy=False), names, sizes


@dataclass(frozen=True)
class ActionChunk:
    """A time-major action chunk together with its physical semantics."""

    values: np.ndarray
    representation: str
    frame: str
    semantics: str
    source: str
    rate_hz: float | None = None
    component_names: tuple[str, ...] = ()
    normalization: str = "none"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    temporal_layout: str = "future"

    def __post_init__(self) -> None:
        if self.temporal_layout not in {"future", "inclusive"}:
            raise ValueError("Unknown action temporal_layout")
        values = _numeric_array(self.values, name="action")
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
            raise ValueError(f"ActionChunk.values must have shape [H,A], got {values.shape}")
        for name, value in (
            ("representation", self.representation),
            ("frame", self.frame),
            ("semantics", self.semantics),
            ("source", self.source),
        ):
            if not value:
                raise ValueError(f"ActionChunk.{name} must be non-empty")
        if self.rate_hz is not None and self.rate_hz <= 0:
            raise ValueError(f"ActionChunk.rate_hz must be positive, got {self.rate_hz}")
        if self.component_names and len(self.component_names) != values.shape[1]:
            raise ValueError(
                f"Action component_names has {len(self.component_names)} entries for "
                f"action width {values.shape[1]}"
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "component_names", tuple(self.component_names))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def horizon(self) -> int:
        return int(self.values.shape[0])

    @property
    def advance_steps(self) -> int:
        return self.horizon - (self.temporal_layout == "inclusive")

    @property
    def dimension(self) -> int:
        return int(self.values.shape[1])

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        representation: str = "unspecified",
        frame: str = "unspecified",
        semantics: str = "unspecified",
        source: str = "adapter",
        rate_hz: float | None = None,
        component_names: tuple[str, ...] = (),
        normalization: str = "none",
        metadata: Mapping[str, Any] | None = None,
        temporal_layout: str = "future",
    ) -> "ActionChunk":
        if isinstance(value, cls):
            return value
        return cls(
            values=value,
            representation=representation,
            frame=frame,
            semantics=semantics,
            source=source,
            rate_hz=rate_hz,
            component_names=component_names,
            normalization=normalization,
            metadata=metadata or {},
            temporal_layout=temporal_layout,
        )


@dataclass(frozen=True)
class ObservationBundle:
    """Named RGB histories, instruction, and typed proprioceptive state."""

    views: Mapping[str, np.ndarray]
    instruction: str
    robot_state: RobotState | None = None
    step_index: int = 0
    timestamp: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.views:
            raise ValueError("ObservationBundle.views must contain at least one named view")
        views = {
            str(key): as_rgb_history(value, name=str(key))
            for key, value in self.views.items()
        }
        if len(views) != len(self.views) or any(not key for key in views):
            raise ValueError("Observation view names must be unique and non-empty")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("ObservationBundle.instruction must be a non-empty string")
        if self.step_index < 0:
            raise ValueError(f"step_index must be non-negative, got {self.step_index}")
        object.__setattr__(self, "views", views)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def require_views(self, names: tuple[str, ...] | list[str]) -> None:
        missing = [name for name in names if name not in self.views]
        if missing:
            raise ValueError(f"Observation is missing required views: {', '.join(missing)}")

    def terminal_view(self, name: str) -> np.ndarray:
        if name not in self.views:
            raise KeyError(f"Unknown observation view {name!r}; available: {sorted(self.views)}")
        return self.views[name][-1].copy()

    def adapter_views(
        self,
        mapping: Mapping[str, str] | None = None,
        *,
        mode: str = "mapping",
        squeeze_single_frame: bool = False,
    ) -> Any:
        """Map canonical view names to one adapter's declared input layout."""
        mapping = dict(mapping or {name: name for name in self.views})
        self.require_views(list(mapping.values()))
        selected = {
            adapter_name: self.views[bundle_name][0].copy()
            if squeeze_single_frame and self.views[bundle_name].shape[0] == 1
            else self.views[bundle_name].copy()
            for adapter_name, bundle_name in mapping.items()
        }
        if mode == "mapping":
            return selected
        values = list(selected.values())
        if mode == "single":
            if len(values) != 1:
                raise ValueError(f"single view mode requires one mapped view, got {len(values)}")
            value = values[0]
            return value[0] if value.ndim == 4 and value.shape[0] == 1 else value
        if mode == "list":
            return values
        if mode == "stack":
            return np.stack(values, axis=0)
        raise ValueError(f"Unsupported adapter view mode {mode!r}")

    def next(
        self,
        *,
        views: Mapping[str, Any],
        robot_state: RobotState | None,
        timestamp: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ObservationBundle":
        return replace(
            self,
            views=views,
            robot_state=robot_state,
            step_index=self.step_index + 1,
            timestamp=timestamp,
            metadata={**self.metadata, **dict(metadata or {})},
        )


@dataclass(frozen=True)
class WorldModelOutput:
    """Normalized output of one world-model action chunk."""

    views: Mapping[str, np.ndarray]
    robot_state: RobotState | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.views:
            raise ValueError("WorldModelOutput.views cannot be empty")
        object.__setattr__(
            self,
            "views",
            {str(key): as_rgb_history(value, name=str(key)) for key, value in self.views.items()},
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def terminal_views(self) -> dict[str, np.ndarray]:
        return {name: history[-1].copy() for name, history in self.views.items()}


__all__ = [
    "ActionChunk",
    "ObservationBundle",
    "RobotState",
    "STATE_SOURCES",
    "WorldModelOutput",
    "as_rgb_history",
]
