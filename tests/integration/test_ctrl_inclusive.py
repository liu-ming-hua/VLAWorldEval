"""Optional native-domain parity; requires separately acquired assets or runtimes."""

import subprocess
from pathlib import Path

import numpy as np
import pytest


def test_actual_ctrl_source_indices_and_no_tail_padding():
    root = Path("datasets/cosmos3_droid/success")
    if not root.is_dir():
        pytest.skip("separately acquired dataset absent")
    pytest.importorskip("pyarrow")
    from evaluation.datasets.droid import load_droid_episode

    data = load_droid_episode(root, 795, 15, 24, protocol="ctrl_world")
    indices = np.asarray(data.provenance["action_source_frame_indices"])
    assert np.array_equal(indices[0], [15, 18, 21, 24, 27])
    assert indices[-1, -1] == 303
    assert data.provenance["end_frame"] == 303
    assert np.array_equal(
        np.stack(data.replay_actions)[:-1, -1], np.stack(data.replay_actions)[1:, 0]
    )
    with pytest.raises(ValueError, match="no tail padding"):
        load_droid_episode(root, 795, 15, 25, protocol="ctrl_world")


def test_camera_vae_protocol_in_optional_ctrl_environment():
    python = Path("environment/world_models/ctrl_world/bin/python")
    if not python.exists():
        pytest.skip("optional Torch runtime absent")
    # Real torch on CPU, stub VAE only. Checks actual adapter methods and RNG.
    code = """
from types import SimpleNamespace
import torch, numpy as np
from adapters.world_models.ctrl_world import CtrlWorldAdapter
wm=object.__new__(CtrlWorldAdapter)
wm.device=torch.device('cpu'); wm.dtype=torch.float32
wm.height=16; wm.width=16; wm.num_frames=5; wm.decode_chunk_size=7
calls=[]
class VAE:
 config=SimpleNamespace(scaling_factor=1.)
 def encode(self,x):
  calls.append(('encode',tuple(x.shape)))
  v=torch.zeros((1,4,2,2))+len(calls)
  return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda:v))
 def decode(self,x,num_frames):
  calls.append(('decode',tuple(x.shape),num_frames))
  return SimpleNamespace(sample=torch.zeros((len(x),3,16,16)))
wm.pipeline=SimpleNamespace(vae=VAE())
latent=wm._encode_views([np.zeros((16,16,3),np.uint8) for _ in range(3)])
assert latent.shape==(1,4,6,2)
assert [x[1] for x in calls]==[(1,3,16,16)]*3
video=wm._decode_latents(torch.zeros((5,3,4,2,2)))
assert video.shape==(5,3,16,16,3)
assert [x[2] for x in calls if x[0]=='decode']==[7,7,1]
wm.num_history=6; wm._memory_latents=[latent.clone() for _ in range(27)]
wm._memory_actions=[np.zeros(7) for _ in range(27)]; wm._current_latent=latent
state=torch.random.get_rng_state().clone()
info=wm.reanchor_image([np.zeros((16,16,3),np.uint8) for _ in range(3)])
assert torch.equal(state,torch.random.get_rng_state()) and info['step_index']==3
assert len(wm._memory_actions)==27
print('camera encode/decode and reanchor verified')
"""
    subprocess.run([str(python), "-c", code], check=True, timeout=60)
