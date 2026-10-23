"""New native integrations remain discoverable without loading model stacks."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from adapters.environment import load_environment_registry
from evaluation.planning.catalog import load_model_catalog
from evaluation.planning.registry import load_runtime_registry, resolve_bindings

pytestmark = pytest.mark.process

ROOT = Path(__file__).resolve().parents[2]
MODELS = (
    ("dreamdojo_2b_gr1", "DreamDojoAdapter", "dreamdojo"),
    ("iws_pusht_cam1", "IWSAdapter", "iws"),
    ("playworld_svd", "PlayWorldAdapter", "playworld"),
    ("dreamtrue_threeview_v1", "DreamTrueAdapter", "dreamtrue"),
    ("irasim_languagetable_frame_ada", "IRASimAdapter", "irasim"),
)


@pytest.mark.parametrize("identifier,adapter,module", MODELS)
def test_catalog_environment_and_constructor_bindings_agree(identifier, adapter, module):
    entry = load_model_catalog(ROOT)["world_models"][identifier]
    definition = load_runtime_registry(ROOT).models["world_models"][adapter]
    environment = f"world_models.{module}"
    assert entry["adapter"] == adapter
    assert entry["environment"] == definition["environment"] == environment
    assert definition["module"] == f"adapters.world_models.{module}"
    assert not definition["distributed"]  # No untested multi-GPU assertion.
    spec = load_environment_registry()[environment]
    assert spec.scope == "world_model"
    assert spec.path == ROOT / "environment/world_models" / module
    assert entry["integration"]["level"] in {"experimental", "native_replay"}
    assert not entry["integration"]["supported_pair_ids"]

    source = ROOT / "adapters/world_models" / f"{module}.py"
    tree = ast.parse(source.read_text())
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == adapter
    )
    constructor = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    parameters = {arg.arg for arg in constructor.args.args + constructor.args.kwonlyargs}
    assert set(definition["kwargs"]) <= parameters
    resolved = resolve_bindings(definition["kwargs"], ROOT, {}, {}, ROOT / "unused")
    assert set(resolved) == set(definition["kwargs"])


@pytest.mark.parametrize("_,adapter,module", MODELS)
def test_lazy_public_exports_do_not_require_model_packages(_, adapter, module):
    code = (
        "import sys; from adapters.world_models import " + adapter + "; "
        "assert " + adapter + ".environment_key == 'world_models." + module + "'; "
        "assert not {'torch','diffusers','transformers','lightning','interactive_world_sim'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True)


def test_native_profiles_do_not_silently_create_policy_pairs():
    registry = load_runtime_registry(ROOT)
    new_adapters = {adapter for _, adapter, _ in MODELS}
    assert not any(profile["world_model"] in new_adapters for profile in registry.pairs)
    for path in (ROOT / "configs/replay").glob("*.json"):
        config = json.loads(path.read_text())
        if config["adapter"] in new_adapters:
            assert (
                config["environment"]
                == registry.models["world_models"][config["adapter"]]["environment"]
            )
