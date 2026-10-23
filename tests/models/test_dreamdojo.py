"""CPU-only protocol checks; these do not claim to run the model."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from adapters.world_models.dreamdojo import DreamDojoAdapter, checkpoint_directory

pytestmark = pytest.mark.process


@pytest.mark.parametrize(
    "features",
    [
        {"dropout_p": 0.1},
        {"alibi_slopes": [1.0]},
        {"softcap": 1.0},
    ],
)
def test_custom_flash_rejects_unsupported_features(features):
    from adapters.world_models.dreamdojo import validate_attention_features

    with pytest.raises(ValueError, match="inference-only Flash"):
        validate_attention_features(**features)


def test_custom_flash_supported_profile():
    from adapters.world_models.dreamdojo import validate_attention_features

    validate_attention_features(dropout_p=0.0, alibi_slopes=None, softcap=0.0)


def test_tokenizer_requires_exact_current_official_bytes():
    import runpy

    import adapters.world_models.dreamdojo as adapter

    root = Path(__file__).resolve().parents[2]
    downloader = runpy.run_path(str(root / "scripts/download/dreamdojo.py"))
    for name in ("TOKENIZER_SHA256", "TOKENIZER_SIZE", "TOKENIZER_OFFICIAL_REVISION"):
        assert downloader[name] == getattr(adapter, name)
    record = {"size": adapter.TOKENIZER_SIZE, "sha256": adapter.TOKENIZER_SHA256}
    manifest = {
        "repository": "Wan-AI/Wan2.1-T2V-1.3B",
        "revision": adapter.TOKENIZER_TRANSPORT_REVISION,
        "files": {"Wan2.1_VAE.pth": record},
    }
    identity = adapter.validate_tokenizer_identity(manifest)
    assert identity["revision"] == adapter.TOKENIZER_OFFICIAL_REVISION
    assert not identity["historical_registry_revision_verified"]
    record["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Tokenizer bytes differ"):
        adapter.validate_tokenizer_identity(manifest)


def test_custom_flash_dispatch_guard_and_restoration(monkeypatch):
    from types import ModuleType

    from adapters.world_models.dreamdojo import restricted_flash_attention

    package = ModuleType("flash_attn")
    interface = ModuleType("flash_attn.flash_attn_interface")
    package.flash_attn_interface = interface
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "forward"

    interface._wrapped_flash_attn_forward = original
    interface._wrapped_flash_attn_varlen_forward = original
    monkeypatch.setitem(sys.modules, "flash_attn", package)
    monkeypatch.setitem(sys.modules, "flash_attn.flash_attn_interface", interface)
    with restricted_flash_attention():
        assert interface._wrapped_flash_attn_forward(None, None, None, 0.0) == "forward"
        assert interface._wrapped_flash_attn_varlen_forward(*([None] * 7), 0.0) == "forward"
        with pytest.raises(ValueError, match="inference-only Flash"):
            interface._wrapped_flash_attn_forward(*([None] * 3), 0.1)
        with pytest.raises(ValueError, match="inference-only Flash"):
            interface._wrapped_flash_attn_varlen_forward(*([None] * 7), 0.0, softcap=1.0)
    assert interface._wrapped_flash_attn_forward is original
    assert interface._wrapped_flash_attn_varlen_forward is original
    assert len(calls) == 2


def test_official_distributed_checkpoint_accepts_empty_unused_ranks(tmp_path):
    iteration = tmp_path / "2B_GR1_post-train/iter_000050000"
    model = iteration / "model"
    model.mkdir(parents=True)
    (model / ".metadata").write_bytes(b"metadata")
    (model / "__0_0.distcp").write_bytes(b"nonempty shard")
    (model / "__100_0.distcp").touch()
    (iteration.parent / "latest_checkpoint.txt").write_text(iteration.name)
    for candidate in (tmp_path, iteration.parent, iteration, model):
        assert checkpoint_directory(candidate) == model


def test_partial_and_unsafe_checkpoint_fail(tmp_path):
    with pytest.raises(FileNotFoundError):
        checkpoint_directory(tmp_path)
    (tmp_path / "latest_checkpoint.txt").write_text("iter_../../elsewhere")
    with pytest.raises(ValueError, match="Unsafe"):
        checkpoint_directory(tmp_path)


def test_pinned_manifest_cannot_omit_a_rank(tmp_path):
    from adapters.world_models.dreamdojo import (
        MODEL_PREFIX,
        MODEL_REVISION,
        verify_pinned_checkpoint,
    )
    from vlaworldeval.assets import MANIFEST, file_record

    names = [f"{MODEL_PREFIX}/.metadata", "2B_GR1_post-train/latest_checkpoint.txt"]
    names += [f"{MODEL_PREFIX}/__{rank}_0.distcp" for rank in range(128)]
    records = {}
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        records[name] = file_record(path)
    marker = {
        "schema_version": 1,
        "repository": "nvidia/DreamDojo",
        "revision": MODEL_REVISION,
        "files": records,
    }
    (tmp_path / MANIFEST).write_text(json.dumps(marker))
    model = tmp_path / MODEL_PREFIX
    assert verify_pinned_checkpoint(tmp_path, model)["ok"]
    (model / "__127_0.distcp").unlink()
    with pytest.raises(ValueError, match="missing or corrupted"):
        verify_pinned_checkpoint(tmp_path, model)
    del records[f"{MODEL_PREFIX}/__127_0.distcp"]
    (tmp_path / MANIFEST).write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="128 model rank shards"):
        verify_pinned_checkpoint(tmp_path, model)


@pytest.mark.parametrize("shape", [(12, 7), (11, 384), (13, 384), (384,)])
def test_does_not_invent_action_mapping_or_padding(shape):
    with pytest.raises(ValueError, match="native GR1"):
        DreamDojoAdapter._actions(np.zeros(shape))


def test_action_validation():
    actions = np.zeros((12, 384), np.float32)
    np.testing.assert_array_equal(DreamDojoAdapter._actions(actions), actions)
    actions[0, 0] = np.nan
    with pytest.raises(ValueError):
        DreamDojoAdapter._actions(actions)
    actions[0, 0] = 0
    actions[0, 30] = 1
    with pytest.raises(ValueError, match="slots"):
        DreamDojoAdapter._actions(actions)


@pytest.mark.parametrize("dtype", [bool, complex, str, object])
def test_actions_reject_lossy_casts(dtype):
    with pytest.raises(ValueError, match="real numeric"):
        DreamDojoAdapter._actions(np.zeros((12, 384), dtype=dtype))


def test_actions_reject_mixed_python_bool_before_numpy_promotion():
    actions = np.zeros((12, 384), dtype=float).tolist()
    actions[0][0] = True
    with pytest.raises(ValueError, match="real numeric"):
        DreamDojoAdapter._actions(actions)


def test_missing_learned_tensor_is_fatal():
    from adapters.world_models.dreamdojo import validate_tensor_coverage

    expected = {"net.action_embedder.weight": (32, 384), "net.output.weight": (16, 32)}
    available = {"net_ema.action_embedder.weight": (32, 384)}
    with pytest.raises(ValueError, match="net_ema.output.weight"):
        validate_tensor_coverage(expected, available)
    available["net_ema.output.weight"] = (16, 31)
    with pytest.raises(ValueError, match="shape_mismatch"):
        validate_tensor_coverage(expected, available)
    available["net_ema.output.weight"] = (16, 32)
    available["optimizer.moment"] = (16, 32)
    assert validate_tensor_coverage(expected, available)["learned_parameters_verified"] == 2


def test_state_dict_hooks_canonicalize_wrappers_without_dropping_parameters():
    from types import SimpleNamespace

    from adapters.world_models.dreamdojo import (
        learned_parameter_inventory,
        validate_tensor_coverage,
    )

    parameter = SimpleNamespace(shape=(2, 3))
    buffer = SimpleNamespace(shape=(2,))

    class WrappedNet:
        def parameters(self):
            return iter([parameter])

        def state_dict(self, *, keep_vars):
            assert keep_vars
            return {"blocks.0.weight": parameter, "buffer": buffer}

    expected = learned_parameter_inventory(WrappedNet())
    assert expected == {"net.blocks.0.weight": (2, 3)}
    with pytest.raises(ValueError, match="net_ema.blocks.0.weight"):
        validate_tensor_coverage(expected, {})

    class BrokenHook(WrappedNet):
        def state_dict(self, *, keep_vars):
            return {"buffer": buffer}

    with pytest.raises(ValueError, match="every learned Parameter"):
        learned_parameter_inventory(BrokenHook())
    other = SimpleNamespace(shape=(2, 3))

    class CollidingHook(WrappedNet):
        def parameters(self):
            return iter([parameter, other])

        def state_dict(self, *, keep_vars):
            # A hook overwrote one canonical key with a different Parameter.
            return {"blocks.0.weight": other}

    with pytest.raises(ValueError, match="every learned Parameter"):
        learned_parameter_inventory(CollidingHook())


def test_auxiliary_encoder_missing_parameter_fails():
    from adapters.world_models.dreamdojo import validate_parameter_shapes

    with pytest.raises(ValueError, match="visual.weight"):
        validate_parameter_shapes({"visual.weight": (2, 3)}, {})


def test_unverified_training_only_lam_cannot_be_used():
    from adapters.world_models.dreamdojo import reject_training_only_lam

    with pytest.raises(RuntimeError, match="must never run"):
        reject_training_only_lam(None, ())


def test_tensor_dtype_rejected_before_any_conversion():
    class BoolTensor:
        dtype = "torch.bool"

        def detach(self):
            raise AssertionError("must reject before conversion")

    with pytest.raises(ValueError, match="real numeric"):
        DreamDojoAdapter._actions(BoolTensor())


def test_reset_clears_rollout_state_and_copies_image():
    adapter = DreamDojoAdapter.__new__(DreamDojoAdapter)
    image = np.zeros((480, 640, 3), np.uint8)
    adapter.reset(image)
    image[:] = 255
    assert not adapter.current_image.any()
    assert adapter.last_video is None and adapter.last_policy_state is None
    assert adapter._chunk_index == 0
    with pytest.raises(ValueError, match="uint8"):
        adapter.reset(np.zeros((480, 640, 3), np.float32))


def test_step_before_reset_fails_without_loading_runtime():
    adapter = DreamDojoAdapter.__new__(DreamDojoAdapter)
    adapter.current_image = None
    with pytest.raises(RuntimeError, match="reset"):
        adapter.step(np.zeros((12, 384)))


def test_import_is_heavy_dependency_free():
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import adapters.world_models.dreamdojo; "
            "assert 'torch' not in sys.modules; assert 'cosmos_predict2' not in sys.modules",
        ],
        cwd=root,
        check=True,
    )


def test_constructor_uses_workspace_asset_root(tmp_path, monkeypatch):
    import adapters.world_models.dreamdojo as module

    source = tmp_path / "third_party/dreamdojo"
    config = source / module.CONFIG
    config.parent.mkdir(parents=True)
    config.touch()
    assets = tmp_path / "checkpoints/world_models/dreamdojo"
    checkpoint = assets / module.MODEL_PREFIX
    monkeypatch.setenv("VLAWORLDEVAL_ROOT", str(tmp_path))
    monkeypatch.setattr(module, "checkpoint_directory", lambda _: checkpoint)
    monkeypatch.setattr(module, "verify_pinned_checkpoint", lambda *_: {"ok": True})
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda command, **_: module.SOURCE_REVISION if command[-1] == "HEAD" else "",
    )
    adapter = module.DreamDojoAdapter(checkpoint_path=assets)
    assert adapter.repo_path == source
    assert adapter.assets_path == assets


def test_native_config_matches_shared_protocol():
    from evaluation.manifests import ActionSpec, VideoSpec

    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "configs/replay/dreamdojo_native_v1.json").read_text())
    action, video = ActionSpec.from_dict(config["action"]), VideoSpec.from_dict(config["video"])
    assert action.dimension == 384 and action.horizon == 12
    assert video.frames_per_chunk == 12 and video.temporal_layout == "future"
    assert config["module"] == "adapters.world_models.dreamdojo"


def test_four_frame_baselines_and_upstream_state_alias():
    from evaluation.datasets.dreamdojo import encode_actions

    values = np.repeat(np.arange(14, dtype=np.float32)[:, None], 29, axis=1)
    state = np.full(29, -0.5, dtype=np.float32)
    actions = encode_actions(values, state, 1)[0]
    np.testing.assert_array_equal(actions[0, :29], state)
    np.testing.assert_array_equal(actions[1:4, 0], [2, 3, 4])
    np.testing.assert_array_equal(actions[4:8, 0], [1, 2, 3, 4])
    assert not actions[:, 29:].any()


def test_native49_global_alias_is_not_reinjected_at_each_round():
    from evaluation.datasets.dreamdojo import encode_actions

    values = np.repeat(np.arange(50, dtype=np.float32)[:, None], 29, axis=1)
    state = np.full(29, -0.5, dtype=np.float32)
    actions = encode_actions(values, state, 4)
    assert actions.shape == (4, 12, 384)
    np.testing.assert_array_equal(actions[0, 0, :29], state)
    # Match global upstream action encoding: later windows begin with deltas,
    # not a future recorded state or the previous command substituted as state.
    np.testing.assert_array_equal(actions[1:, 0, :29], np.ones((3, 29)))
    assert not actions[..., 29:].any()


def test_multi_chunk_encoding_and_stepping_explicitly_refused():
    from evaluation.datasets.dreamdojo import encode_actions

    with pytest.raises(ValueError, match="multi-chunk"):
        encode_actions(np.zeros((26, 29)), np.zeros(29), 2)
    adapter = DreamDojoAdapter.__new__(DreamDojoAdapter)
    adapter.current_image = np.zeros((480, 640, 3), np.uint8)
    adapter._chunk_index = 1
    with pytest.raises(RuntimeError, match="multi-chunk"):
        adapter.step(np.zeros((12, 384)))


def test_replay_rejects_multi_chunk_flag_even_when_packet_already_exists():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "scripts/replay/dreamdojo.py", "--chunks", "2", "--admit-only"],
        cwd=root,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "multi-chunk replay is unsupported" in result.stderr


def test_superseded_episode_stats_packet_is_refused():
    from evaluation.datasets.dreamdojo import validate_prepared_packet

    with pytest.raises(ValueError, match="superseded"):
        validate_prepared_packet({"source": {}, "actions": np.zeros((1, 12, 384))})


def test_downloader_disables_partial_deleting_backend_and_all_proxy_casings(monkeypatch):
    import os
    import runpy

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
    monkeypatch.setenv("HttP_PrOxY", "http://invalid.example")
    module = runpy.run_path(str(root / "scripts/download/dreamdojo.py"))
    module["configure_network"]()
    assert os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "0"
    assert not any(key.lower().endswith("_proxy") for key in os.environ)


def test_replay_does_not_publish_success_evidence_for_failed_cleanup(tmp_path):
    import runpy

    root = Path(__file__).resolve().parents[2]
    replay = runpy.run_path(str(root / "scripts/replay/dreamdojo.py"))
    with pytest.raises(RuntimeError, match="did not complete"):
        replay["record_compatibility_evidence"](
            tmp_path, {"status": "failed", "world_model_calls": 1}
        )
    assert not (tmp_path / "compatibility_profile.json").exists()


def test_replay_reads_durable_console_evidence_not_consumed_rpc_packets(tmp_path):
    import runpy

    from adapters.world_models.dreamdojo import COMPATIBILITY_PROFILE

    root = Path(__file__).resolve().parents[2]
    replay = runpy.run_path(str(root / "scripts/replay/dreamdojo.py"))
    worker = tmp_path / "worker"
    worker.mkdir()
    metadata = {
        "compatibility_profile": COMPATIBILITY_PROFILE,
        "asset_fidelity_status": "unit_fixture",
    }
    (worker / "console.log").write_text("DREAMDOJO_STEP_EVIDENCE=" + json.dumps(metadata) + "\n")
    replay["record_compatibility_evidence"](
        tmp_path, {"status": "completed", "world_model_calls": 1}
    )
    result = json.loads((tmp_path / "compatibility_profile.json").read_text())
    assert result["steps"][0]["metadata"] == metadata
    assert result["steps"][0]["evidence_source"] == "worker/console.log"


def test_native_normalization_keeps_constant_joints_zero():
    from evaluation.datasets.dreamdojo import PARTS, normalize_parts

    dims = [7, 7, 6, 6, 3]
    metadata = {}
    start = 0
    for part, dim in zip(PARTS, dims):
        metadata[part] = {"original_key": "action", "start": start, "end": start + dim}
        start += dim
    low = np.zeros(29, np.float32)
    high = np.ones(29, np.float32)
    high[1] = 0
    result = normalize_parts(
        {"action": np.full((2, 29), 0.5)}, metadata, {"action": {"min": low, "max": high}}
    )
    assert result.shape == (2, 29) and not result.any()


def test_default_stats_layout_is_distinct_from_raw_episode_layout():
    from evaluation.datasets.dreamdojo import PARTS, normalize_parts

    raw, compact = {}, {}
    offset = 0
    for part, size in zip(PARTS, [7, 7, 6, 6, 3]):
        raw[part] = {"original_key": "action", "start": offset + 100, "end": offset + 100 + size}
        compact[part] = {"original_key": "action", "start": offset, "end": offset + size}
        offset += size
    result = normalize_parts(
        {"action": np.full((14, 150), 0.5)},
        raw,
        {"action": {"min": np.zeros(29), "max": np.ones(29)}},
        compact,
    )
    assert result.shape == (14, 29) and not result.any()
