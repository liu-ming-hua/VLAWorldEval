"""Opt-in first-window parity against the local upstream forward_wm method.

Imported only inside the Ctrl Torch worker. Does not execute upstream main,
load another checkpoint, download data, or use GT frames as model predictions.
"""
import ast
import hashlib
import json
from types import MethodType, SimpleNamespace

import numpy as np


def verify_ctrl_reference(adapter, action, first_action):
    import einops
    import torch
    if type(adapter).__module__ != "adapters.world_models.ctrl_world":
        raise ValueError("Reference audit is specific to CtrlWorldAdapter")
    if len(adapter._memory_latents) != 24:
        raise ValueError("Reference parity requires an untouched reset worker")
    path = adapter.repo_path / "scripts/rollout_replay_traj.py"
    source = path.read_text()
    cls = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "agent")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in {"forward_wm", "normalize_bound"}]
    if len(methods) != 2:
        raise ValueError("Pinned upstream reference methods changed")
    namespace = {"np": np, "torch": torch, "einops": einops, "CtrlWorldDiffusionPipeline": adapter._pipeline_cls}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    stats = json.loads(adapter.data_stat_path.read_text())
    ref = SimpleNamespace(model=adapter.model, device=adapter.device, dtype=adapter.dtype,
        state_p01=np.array(stats["state_01"])[None], state_p99=np.array(stats["state_99"])[None],
        args=SimpleNamespace(num_frames=5, num_history=6, action_dim=7, width=adapter.width,
            height=adapter.height, num_inference_steps=adapter.num_inference_steps,
            decode_chunk_size=adapter.decode_chunk_size, guidance_scale=adapter.guidance_scale,
            fps=adapter.fps, motion_bucket_id=adapter.motion_bucket_id))
    ref.normalize_bound = MethodType(namespace["normalize_bound"], ref)
    history = torch.cat([adapter._memory_latents[i] for i in adapter._official_history_indices]).unsqueeze(0)
    action_cond = np.concatenate([np.repeat(np.asarray(first_action)[None], 6, axis=0), np.asarray(action)])
    latent = adapter._current_latent
    views = torch.chunk(latent[0], 3, dim=1)
    truth = [view[None].repeat(5, 1, 1, 1) for view in views]
    devices = [adapter.device.index or 0] if adapter.device.type == "cuda" else []
    # Reference is the actual upstream AST method; restore RNG before adapter
    # generation. Ground-truth decode in forward_wm is display-only, not input.
    with torch.inference_mode(), torch.random.fork_rng(devices=devices):
        _, _, reference_video, reference_latents = namespace["forward_wm"](
            ref, action_cond, truth, latent, his_cond=history, text=adapter.prompt)
    with torch.inference_mode():
        adapter.step(action)
    ref_video = reference_video.transpose(1, 0, 2, 3, 4)
    ref_latents = reference_latents.permute(1, 0, 2, 3, 4)
    delta = np.abs(adapter.last_video.astype(np.int16) - ref_video.astype(np.int16))
    latent_delta = (adapter.last_latents - ref_latents).abs()
    return {"status": "passed" if torch.equal(adapter.last_latents, ref_latents) and np.array_equal(adapter.last_video, ref_video) else "not_exact",
        "world_model_calls": 2, "vla_calls": 0, "reference": str(path),
        "reference_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "latent_exact": torch.equal(adapter.last_latents, ref_latents),
        "pixel_exact": bool(np.array_equal(adapter.last_video, ref_video)),
        "max_latent_difference": float(latent_delta.max()), "max_pixel_difference": int(delta.max()),
        "mean_pixel_difference": float(delta.mean()), "video_shape": list(ref_video.shape),
        "scope": "same initial latent/history, actual upstream forward_wm, same RNG; not full upstream dataset-loader parity",
        "task_success_evaluated": False}
