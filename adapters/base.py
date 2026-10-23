from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import torch


class VLAAdapter(ABC):
    """Embodiment/model-specific VLA interface.

    ``environment_key`` points to the central model-environment registry.
    Adapters return a time-major action chunk. ``state`` is optional so models
    that need proprioception can be added without changing the evaluator.
    """

    environment_key = "unknown"

    def environment_spec(self):
        from adapters.environment import adapter_environment

        return adapter_environment(self)

    @abstractmethod
    def reset(self, initial_state: Any = None) -> None:
        pass

    @abstractmethod
    def predict(self, image: Any, instruction: str, state: Any = None) -> torch.Tensor:
        """Return a time-major ``[H, A]`` action chunk.

        ``image`` may be one RGB array or a mapping of named camera streams.
        Model-specific adapters validate the exact camera, history, state, and
        action protocol instead of silently inventing missing observations.
        """
        pass

    def update_state(self, state: Any, action: torch.Tensor, observation: np.ndarray) -> Any:
        return state

    def predict_observation(self, observation: Any, adapter_spec: Any) -> Any:
        """Run the legacy ``predict`` method from a typed observation bundle."""
        image = adapter_spec.adapter_input(observation)
        state = None
        if observation.robot_state is not None:
            state = observation.robot_state.as_policy_input()
        return self.predict(image, observation.instruction, state)


class WorldModelAdapter(ABC):
    """Action-conditioned environment interface used by the evaluator."""

    environment_key = "unknown"

    def environment_spec(self):
        from adapters.environment import adapter_environment

        return adapter_environment(self)

    @abstractmethod
    def reset(self, image: np.ndarray) -> None:
        pass

    @abstractmethod
    def step(self, action: torch.Tensor) -> np.ndarray:
        """Apply an action chunk and return the terminal generated image."""
        pass

    def reanchor_image(self, image: Any) -> dict:
        """Diagnostic visual intervention, not a policy reset or state update.

        Unsupported models fail explicitly. Normal closed-loop evaluation never
        calls this hook; recorded future images must not enter benchmark runs.
        """
        raise NotImplementedError(f"{type(self).__name__} has no visual reanchor protocol")

    def step_output(
        self,
        action: Any,
        *,
        output_views: tuple[str, ...] = (),
        state_schema: str | None = None,
    ) -> Any:
        """Normalize a legacy adapter step into ``WorldModelOutput``.

        Model adapters may override this when their output is not represented by
        ``step`` plus ``last_video``/``last_policy_state``.  This compatibility
        implementation keeps existing adapters usable while making output
        provenance explicit to the evaluator.
        """
        import numpy as np

        from evaluation.contracts import RobotState, WorldModelOutput

        terminal = self.step(action)
        video = getattr(self, "last_video", None)
        value = np.asarray(video if video is not None else terminal)
        names = tuple(output_views) or tuple(getattr(self, "view_names", ()))

        if value.ndim == 3:
            names = names or ("primary",)
            if len(names) != 1:
                raise ValueError(
                    f"World model returned one RGB view but declared {len(names)} output views"
                )
            views = {names[0]: value}
        elif value.ndim == 4:
            names = names or ("primary",)
            if len(names) != 1:
                raise ValueError(
                    f"World model returned one video stream but declared {len(names)} output views"
                )
            views = {names[0]: value}
        elif value.ndim == 5 and value.shape[-1] == 3:
            view_count = value.shape[1]
            if not names:
                names = tuple(f"view_{index}" for index in range(view_count))
            if len(names) != view_count:
                raise ValueError(
                    f"World-model video contains {view_count} views but names={names}"
                )
            views = {name: value[:, index] for index, name in enumerate(names)}
        else:
            raise ValueError(
                "World-model output must be RGB [H,W,3], [T,H,W,3], or "
                f"[T,V,H,W,3], got {value.shape}"
            )

        policy_state = getattr(self, "last_policy_state", None)
        robot_state = None
        if policy_state is not None:
            if not state_schema:
                raise ValueError(
                    "World model returned last_policy_state but no state_schema was declared"
                )
            robot_state = RobotState.from_value(
                policy_state,
                schema=state_schema,
                source="world_model_prediction",
            )
        return WorldModelOutput(
            views=views,
            robot_state=robot_state,
            metadata={"adapter": f"{type(self).__module__}.{type(self).__name__}"},
        )
