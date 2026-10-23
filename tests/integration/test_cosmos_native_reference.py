"""Optional native-domain parity; requires separately acquired assets or runtimes."""

import json
from pathlib import Path

import numpy as np
import pytest


def test_native_composition_matches_real_upstream_method():
    pytest.importorskip(
        "torch", reason="Native image preprocessing is exercised in the Cosmos model environment"
    )
    from evaluation.datasets.cosmos_reference import compare_source_compositor, compose_native_views

    root = Path(__file__).resolve().parents[2]
    source = root / "third_party/cosmos-framework"
    if not source.is_dir():
        pytest.skip("Optional upstream runtime is not part of the core distribution")
    rng = np.random.default_rng(0)
    views = {
        f"droid.{name}": rng.integers(0, 256, (2, 24, 32, 3), dtype=np.uint8)
        for name in ("wrist", "exterior_1", "exterior_2")
    }
    composite = compose_native_views(views)
    assert composite.shape == (2, 36, 32, 3)
    assert compare_source_compositor(views, composite, source)["exact_equal"]


def test_reference_crop_matches_actual_upstream_latent_unpadding():
    """Check coordinates, not just dimensions; a center crop would be wrong."""
    import ast
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[2]
    source = (
        root / "third_party/cosmos-framework/cosmos_framework/model/generator/omni_mot_model.py"
    )
    checkpoint = (
        root
        / "checkpoints/world_models/cosmos3_nano_droid_fd/community_jere_mybao_v1_20k_ema/config.json"
    )
    if not source.is_file() or not checkpoint.is_file():
        pytest.skip("Optional upstream source and frozen checkpoint config required")
    factor = json.loads(checkpoint.read_text())["model"]["config"]["tokenizer"][
        "spatial_compression_factor"
    ]
    assert factor == 16
    tree = ast.parse(source.read_text())
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OmniMoTModel"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_remove_padding_from_latent"
    )
    namespace = {"torch": torch, "log": SimpleNamespace(warning=lambda *_: None)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    context = SimpleNamespace(
        tokenizer_vision_gen=SimpleNamespace(spatial_compression_factor=factor)
    )
    latent = torch.arange(34 * 46).reshape(1, 1, 1, 34, 46)
    for shape in ((4,), (1, 4)):
        image_size = torch.tensor([544, 736, 540, 640]).reshape(shape)
        actual = namespace[method.name](context, [latent], [image_size])[0]
        assert torch.equal(actual, latent[..., :33, :40])
        assert tuple(dimension * factor for dimension in actual.shape[-2:]) == (528, 640)
