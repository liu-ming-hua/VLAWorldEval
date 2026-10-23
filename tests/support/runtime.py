"""CPU-only adapters used to test the actual subprocess protocol."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from adapters.base import VLAAdapter, WorldModelAdapter


class FixtureVLA(VLAAdapter):
    def __init__(self):
        self.calls = 0

    def reset(self, initial_state=None):
        self.calls = 0

    def predict(self, image, instruction, state=None):
        self.calls += 1
        value = float(np.asarray(image["camera.front"])[0, 0, 0]) + self.calls
        return np.full((2, 2), value, dtype=np.float32)


class FixtureWM(WorldModelAdapter):
    def __init__(self):
        self.calls = 0

    def reset(self, image):
        self.calls = 0

    def step(self, action):
        self.calls += 1
        value = float(np.asarray(action)[0, 0])
        if value < 0:
            raise ValueError("deliberate fixture failure")
        self.last_video = np.full((2, 2, 16, 16, 3), int(value) % 255, dtype=np.uint8)
        return self.last_video[-1, 0]


class ReusableFixtureWM(WorldModelAdapter):
    supports_episode_reuse = True

    def __init__(self, runtime_dir):
        self.runtime_dir = runtime_dir
        self.calls = 0

    def set_episode_context(self, runtime_dir):
        self.runtime_dir = runtime_dir

    def reset(self, image):
        self.calls = 0
        self.image = np.asarray(image).copy()

    def step(self, action):
        self.calls += 1
        value = int(np.random.randint(20)) + self.calls + int(self.image[0, 0, 0])
        self.last_video = np.full((2, 16, 16, 3), value, np.uint8)
        return self.last_video[-1]


class NoReuseFixtureWM(ReusableFixtureWM):
    supports_episode_reuse = False


class SlowFixtureVLA(FixtureVLA):
    def __init__(self, delay=0, marker=None, ignore_term=False):
        super().__init__()
        if marker:
            Path(marker).write_text(str(os.getpid()))
        if ignore_term:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(delay)

    def predict(self, image, instruction, state=None):
        if instruction == "exit":
            os._exit(23)
        if instruction == "oom":
            raise OutOfMemoryError("synthetic OOM; no allocation")
        time.sleep(30)


class FeedbackFixtureWM(FixtureWM):
    def reanchor_image(self, image):
        assert self.calls > 0
        assert np.asarray(image).shape == (16, 16, 3)
        return {"step_index": self.calls, "state_changed": False}


class InclusiveFixtureWM(FeedbackFixtureWM):
    def step(self, action):
        self.calls += 1
        self.last_video = np.stack(
            [np.full((2, 16, 16, 3), self.calls * 20 + i, np.uint8) for i in range(5)]
        )
        return self.last_video[-1, 0]


class OutOfMemoryError(RuntimeError):
    pass


class BrokenCloseFixtureVLA(FixtureVLA):
    def close(self):
        raise ValueError("deliberate adapter cleanup failure")


class BrokenCloseFixtureWM(FixtureWM):
    def close(self):
        raise ValueError("deliberate world-model cleanup failure")


class StartupFailureVLA(FixtureVLA):
    def __init__(self):
        raise ValueError("deliberate startup failure")


class ChildFixtureVLA(FixtureVLA):
    def __init__(self, marker):
        super().__init__()
        # Same session/group as our worker; no unrelated processes are targeted.
        self.child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        Path(marker).write_text(str(self.child.pid))

    def predict(self, image, instruction, state=None):
        os._exit(24)  # Leave child behind to exercise dead-launcher cleanup.


class CleanLauncherLeakingChildVLA(ChildFixtureVLA):
    """Normal worker exit leaves the child in its process group for cleanup."""

    def predict(self, image, instruction, state=None):
        return np.zeros((2, 2), dtype=np.float32)
