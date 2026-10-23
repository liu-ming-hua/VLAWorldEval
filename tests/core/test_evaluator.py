import json

import numpy as np
from PIL import Image

from adapters.base import VLAAdapter, WorldModelAdapter
from evaluation.evaluator import PolicyWorldModelEvaluator


class FakeVLA(VLAAdapter):
    environment_key = "test.vla"

    def reset(self, initial_state=None):
        self.reset_state = initial_state

    def predict(self, image, instruction, state=None):
        assert image.shape == (256, 256, 3)
        assert instruction == "move the block"
        return np.full((2, 3), len(self.seen), dtype=np.float32)

    def update_state(self, state, action, observation):
        self.seen.append(observation.mean())
        return observation.mean()

    def __init__(self):
        self.seen = []


class FakeWorldModel(WorldModelAdapter):
    environment_key = "test.world_model"

    def reset(self, image):
        self.reset_input = image
        self.step_count = 0

    def step(self, action):
        self.step_count += 1
        return np.full((256, 256, 3), self.step_count, dtype=np.uint8)

    def close(self):
        self.closed = True


def test_rollout_initializes_policy_observation_and_closes_world_model(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    image_path = task_dir / "trial.png"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(image_path)
    (task_dir / "trial.json").write_text(
        json.dumps({"instruction": "move the block"}), encoding="utf-8"
    )

    vla = FakeVLA()
    world_model = FakeWorldModel()
    evaluator = PolicyWorldModelEvaluator(vla, world_model)
    trial = {
        "trial_png": str(image_path),
        "instruction": "move the block",
        "task_key": "task",
        "task_display": "Task",
    }

    video, state, actions = evaluator.rollout(trial, rollout_length=2, return_actions=True)

    assert video.shape == (3, 256, 256, 3)
    assert len(actions) == 2
    assert state == 2.0
    assert vla.reset_state is None
    assert world_model.reset_input.shape == (256, 256, 3)

    evaluator.evaluate(str(tmp_path), rollout_length=1)
    assert world_model.closed is True
