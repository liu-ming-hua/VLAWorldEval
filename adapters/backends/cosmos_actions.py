"""Official RoboLab DROID preprocessing shared by joint-space policy and FD."""
from types import SimpleNamespace

import numpy as np


def validate_joint_action(value, horizon):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (horizon, 8) or not np.isfinite(value).all():
        raise ValueError(f"Require finite absolute DROID actions [{horizon},8]")
    if np.any((value[:, -1] < 0) | (value[:, -1] > 1)):
        raise ValueError("DROID closed-gripper commands must be in [0,1]")
    return value


def build_joint_sample(observation, *, horizon=32, mode="wam", targets=None,
                       resolution="480", format_prompt_as_json=False):
    import torch
    from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyConfig, RobolabPolicyService
    from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline
    if mode not in {"wam", "forward_dynamics"}:
        raise ValueError("Only DROID policy or joint forward dynamics is admitted")
    # Reuse upstream image composition, state row and gripper conversion.
    # Capture BEFORE its transform so FD gets the correct mode/prompt/masks.
    context = SimpleNamespace(cfg=RobolabPolicyConfig(
        checkpoint_path="local", domain_name="droid_lerobot", decode_video=False,
        seed=0, deterministic_seed=True, guidance=1.0, guidance_interval=None,
        num_steps=4, shift=5.0, conditioning_fps=15.0, resolution=resolution,
        action_chunk_size=horizon, action_dim=8, use_state=True, history_length=1),
        _transform=lambda sample, resolution: sample)
    sample = RobolabPolicyService._build_sample(context, observation)
    sample["mode"] = mode
    if mode == "forward_dynamics":
        commands = validate_joint_action(targets, horizon)
        encoded = commands.copy()
        encoded[:, -1] = 1.0 - encoded[:, -1]
        sample["action"][1:] = torch.from_numpy(encoded)
    elif targets is not None:
        raise ValueError("Policy mode must not consume future actions")
    transform = ActionTransformPipeline(max_action_dim=64, cfg_dropout_rate=0.0,
                                       format_prompt_as_json=format_prompt_as_json)
    sample = transform(sample, resolution)
    if isinstance(sample.get("ai_caption"), dict):
        import json
        sample["ai_caption"] = json.dumps(sample["ai_caption"])
    return sample


def generate_joint_sample(runtime, sample, seed):
    import torch
    from cosmos_framework.scripts.action_policy_server_robolab import _build_data_batch_from_sample
    batch = _build_data_batch_from_sample(sample)
    with torch.inference_mode():
        return runtime._pipe.model.generate_samples_from_batch(
            batch, guidance=runtime.guidance, seed=[seed],
            num_steps=runtime.num_inference_steps, shift=runtime.shift)


def decode_joint_video(runtime, generated):
    import torch
    with torch.inference_mode():
        value = runtime._pipe.model.decode(generated["vision"][0])[0]
        return ((value.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()


def restore_robolab_geometry(video, height=540, width=640):
    """Undo latent padding removal without stretching camera boundaries.

    The official model removes padding at latent (16-pixel) granularity. Missing
    bottom/right borders are edge-filled, not represented as predicted pixels.
    """
    video = np.asarray(video)
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError("Require decoded RGB [T,H,W,3]")
    h, w = video.shape[1:3]
    if (h, w) == (height, width):
        return video
    if (h, w) != (height // 16 * 16, width // 16 * 16):
        raise ValueError(f"Unexpected RoboLab decoded geometry {(h, w)}")
    return np.pad(video, ((0, 0), (0, height-h), (0, width-w), (0, 0)), mode="edge")
