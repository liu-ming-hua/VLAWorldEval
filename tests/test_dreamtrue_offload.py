"""Weight-free CPU module traversal fixture; not a full model or CUDA test."""
from types import SimpleNamespace

from adapters.world_models.dreamtrue import complete_offload_map, wrap_camera_encoder


def test_camera_module_traversal_covers_all_parameterized_leaves():
    class Conv3d:
        has_parameters = True

    class GroupNorm:
        has_parameters = True

    class SiLU:
        has_parameters = False

    class Wrapper:
        pass

    original = {Conv3d: Wrapper}
    mapping = complete_offload_map(original, group_norm=GroupNorm, wrapped_module=Wrapper)
    leaves = [Conv3d(), GroupNorm(), SiLU()]
    uncovered = [type(layer).__name__ for layer in leaves
                 if layer.has_parameters and not any(isinstance(layer, cls) for cls in mapping)]
    assert uncovered == []
    assert GroupNorm not in original


def test_camera_lifecycle_never_splits_conv_and_norm_devices():
    class Conv3d:
        pass

    class GroupNorm:
        pass

    class Wrapper:
        pass

    class Encoder:
        def to(self, *, device, dtype):
            self.device, self.dtype = device, dtype
            return self

    calls = []
    def enable(encoder, mapping, *, vram_config):
        calls.append((encoder, mapping, vram_config))
        return encoder

    torch = SimpleNamespace(nn=SimpleNamespace(Conv3d=Conv3d, GroupNorm=GroupNorm), bfloat16="bf16_fixture")
    encoder = Encoder()
    result = wrap_camera_encoder(encoder, device="cuda:0", torch=torch,
                                enable_vram_management=enable, wrapped_module=Wrapper)
    assert result is encoder
    _, mapping, config = calls[0]
    assert mapping[Conv3d] is Wrapper and mapping[GroupNorm] is Wrapper
    assert all(config[f"{stage}_device"] == "cuda:0" for stage in ("offload", "onload", "preparing", "computation"))
    assert all(config[f"{stage}_dtype"] == "bf16_fixture" for stage in ("offload", "onload", "preparing", "computation"))
