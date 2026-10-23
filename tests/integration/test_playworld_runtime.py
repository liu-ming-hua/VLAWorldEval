"""Optional CPU probes of the pinned official source, launched from core.

No world-model weights or GPU are loaded. Missing dedicated environments skip
these probes; portable contract tests remain in test_playworld_adapter.py.
"""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "environment/world_models/playworld/bin/python"


@pytest.mark.skipif(not PYTHON.is_file(), reason="PlayWorld dedicated environment not installed")
def test_official_svd_preparation_stride_history_and_cpu_imports():
    code = """
import sys
sys.path.insert(0, "third_party/playworld")
import numpy as np
import torch
from openworld.world_models.vidwm_world_model import VidWMConfig, VidWMWorldModel
import vidwm.video_models.vidwm_diffusion
cfg = VidWMConfig(device="cpu", action_downsample=3, action_normalize=False)
wm = VidWMWorldModel(cfg)
chunk = np.arange(105, dtype=np.float32).reshape(15,7)
state = {"_initial_robot_state": np.arange(7, dtype=np.float32)}
buffer = wm._get_or_init_state_buffer(state, chunk)
assert len(buffer) == 12
prepared, last = wm._prepare_actions(chunk, 1, state_buffer=buffer)
assert prepared.shape == (1,11,7)
np.testing.assert_array_equal(prepared[0,6:].float().numpy(), chunk[[0,3,6,9,12]])
np.testing.assert_array_equal(last, chunk[12])
np.testing.assert_array_equal(prepared[0,:6].float().numpy(), np.repeat(state["_initial_robot_state"][None],6,axis=0))
latent = torch.zeros(1,4,72,40)
current, history, memory = wm._unpack_state({"current_latent":latent})
assert history.shape == (1,6,4,72,40) and len(memory) == 12
assert cfg.history_idx == (0,0,-12,-9,-6,-3)
assert not torch.cuda.is_initialized()
print("official CPU protocol OK; sampled offsets 0,3,6,9,12; no model inference or CUDA")
"""
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        USE_TF="0",
        HF_HOME=str(ROOT / ".cache/huggingface"),
        XDG_CACHE_HOME=str(ROOT / ".cache"),
    )
    for key in list(env):
        if key.lower().endswith("_proxy"):
            env.pop(key)
    result = subprocess.run(
        [str(PYTHON), "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 0, result.stderr
    assert "official CPU protocol OK" in result.stdout


@pytest.mark.skipif(
    not PYTHON.is_file()
    or not (ROOT / "datasets/playworld/benchmark/irom_test_carrot/init_0").is_dir(),
    reason="Official benchmark/dedicated environment not installed",
)
def test_official_benchmark_packet_is_explicit_probe_and_keeps_pixels(tmp_path):
    code = """
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from evaluation.datasets.playworld import prepare_benchmark_packet
from evaluation.execution.transport import read_packet
root = Path.cwd()
case = root / "datasets/playworld/benchmark/irom_test_carrot/init_0"
out = Path(sys.argv[1]) / "probe.npz"
cfg = root / "configs/replay/playworld_native_v1.json"
for kwargs in ({}, {"hold_probe":True,"actions_path":out}):
 try:
  prepare_benchmark_packet(cfg,case,out,**kwargs)
 except ValueError:
  pass
 else:
  raise AssertionError("must explicitly select exactly one action source")
report = prepare_benchmark_packet(cfg,case,out,hold_probe=True)
packet = read_packet(out)
assert report["model_inference_performed"] is False
assert packet["actions"].shape == (1,15,7)
assert packet["source"]["recorded_action_replay"] is False
assert packet["source"]["sampled_action_offsets"] == [0,3,6,9,12]
assert packet["source"]["source_advance_ticks"] == 15
for name, frame in packet["initial_views"].items():
 np.testing.assert_array_equal(frame,np.asarray(Image.open(case/f"{name}.png").convert("RGB")))
np.testing.assert_array_equal(packet["actions"][0],np.repeat(packet["reset_args"]["value"]["initial_state"][None],15,0))
assert "torch" not in sys.modules
print("official benchmark packet OK; no weights or torch")
"""
    result = subprocess.run(
        [str(PYTHON), "-c", code, str(tmp_path)],
        cwd=ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "official benchmark packet OK" in result.stdout
