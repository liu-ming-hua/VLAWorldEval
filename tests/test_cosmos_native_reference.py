"""Native evaluation preserves decoder pixels; the existing policy bridge is unchanged."""
from pathlib import Path
import json

import numpy as np
import pytest

from adapters.world_models.cosmos3_droid_native import Cosmos3DroidFDNativeAdapter
from adapters.world_models.cosmos3_nano_droid_fd import Cosmos3NanoDroidFDAdapter


def test_native_output_never_edge_fills_existing_policy_geometry(monkeypatch):
    from PIL import Image
    adapter = object.__new__(Cosmos3DroidFDNativeAdapter)
    adapter.current_image = Image.fromarray(np.zeros((540, 640, 3), np.uint8))

    def step(self, action):
        self.last_video = np.zeros((16, 528, 640, 3), np.uint8)
        return self.last_video[-1]

    monkeypatch.setattr(Cosmos3NanoDroidFDAdapter, "step", step)
    result = adapter.step_output(np.zeros((16, 10)), output_views=("droid.cosmos_composite",))
    assert result.views["droid.cosmos_composite"].shape == (16, 528, 640, 3)
    assert result.metadata["conditioning_image_wh"] == [640, 540]
    assert result.metadata["edge_filled_pixels"] is False
    assert result.metadata["pixel_transport"] == "decoded_upstream_mp4_before_framework_video_encoding"
    assert Cosmos3NanoDroidFDAdapter.step_output is not Cosmos3DroidFDNativeAdapter.step_output


def test_native_composition_matches_real_upstream_method():
    pytest.importorskip("torch", reason="Native image preprocessing is exercised in the Cosmos model environment")
    from evaluation.datasets.cosmos_reference import compose_native_views, compare_source_compositor
    root = Path(__file__).resolve().parents[1]
    source = root/"third_party/cosmos-framework"
    if not source.is_dir():
        pytest.skip("Optional upstream runtime is not part of the core distribution")
    rng = np.random.default_rng(0)
    views = {f"droid.{name}": rng.integers(0, 256, (2, 24, 32, 3), dtype=np.uint8)
             for name in ("wrist", "exterior_1", "exterior_2")}
    composite = compose_native_views(views)
    assert composite.shape == (2, 36, 32, 3)
    assert compare_source_compositor(views, composite, source)["exact_equal"]


def test_reference_crop_matches_actual_upstream_latent_unpadding():
    """Check coordinates, not just dimensions; a center crop would be wrong."""
    import ast
    from types import SimpleNamespace
    torch = pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[1]
    source = root/"third_party/cosmos-framework/cosmos_framework/model/generator/omni_mot_model.py"
    checkpoint = root/"checkpoints/world_models/cosmos3_nano_droid_fd/community_jere_mybao_v1_20k_ema/config.json"
    if not source.is_file() or not checkpoint.is_file():
        pytest.skip("Optional upstream source and frozen checkpoint config required")
    factor = json.loads(checkpoint.read_text())["model"]["config"]["tokenizer"]["spatial_compression_factor"]
    assert factor == 16
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OmniMoTModel")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_remove_padding_from_latent")
    namespace = {"torch": torch, "log": SimpleNamespace(warning=lambda *_: None)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    context = SimpleNamespace(tokenizer_vision_gen=SimpleNamespace(spatial_compression_factor=factor))
    latent = torch.arange(34*46).reshape(1, 1, 1, 34, 46)
    for shape in ((4,), (1, 4)):
        image_size = torch.tensor([544, 736, 540, 640]).reshape(shape)
        actual = namespace[method.name](context, [latent], [image_size])[0]
        assert torch.equal(actual, latent[..., :33, :40])
        assert tuple(dimension*factor for dimension in actual.shape[-2:]) == (528, 640)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "bogus"])
def test_explicit_chunk_count_rejects_ambiguous_values(value):
    from evaluation.datasets.cosmos_reference import resolve_chunks
    with pytest.raises(ValueError):
        resolve_chunks(Path("unused"), 0, 0, value)


@pytest.mark.parametrize("length,start,expected", [(34, 0, 2), (34, 1, 2), (34, 2, 1), (17, 0, 1)])
def test_all_chunks_resolves_actual_episode_length_without_padding(tmp_path, length, start, expected):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa
    from evaluation.datasets.cosmos_reference import resolve_chunks
    path = tmp_path/"meta/episodes/chunk-000"
    path.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"episode_index": 7, "length": length}]), path/"file-000.parquet")
    assert resolve_chunks(tmp_path, 7, start, "all") == expected


@pytest.mark.parametrize("rows", [[], [{"episode_index": 7, "length": 16}],
                                  [{"episode_index": 7, "length": 33}]*2])
def test_all_chunks_rejects_missing_duplicate_and_incomplete_episode(tmp_path, rows):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa
    from evaluation.datasets.cosmos_reference import resolve_chunks
    path = tmp_path/"meta/episodes/chunk-000"
    path.mkdir(parents=True)
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), path/"file-000.parquet")
    with pytest.raises(ValueError):
        resolve_chunks(tmp_path, 7, 0, "all")


def test_snapshot_checksum_cache_is_process_local_and_mutation_sensitive(tmp_path, monkeypatch):
    from evaluation.datasets import cosmos_reference as module
    from vlaworldeval.assets import MANIFEST, file_record
    (tmp_path/"sample").write_bytes(b"original")
    manifest = {"schema_version": 1, "repository": "nvidia/Cosmos3-DROID", "revision": "a"*40,
                "files": {"sample": file_record(tmp_path/"sample")}}
    (tmp_path/MANIFEST).write_text(json.dumps(manifest))
    checks = module.SnapshotChecks(tmp_path)
    signature = checks.signature
    monkeypatch.setattr(checks, "signature", lambda p: (*signature(p)[:3], 0, 0))
    actual, calls = module.verify_snapshot, []

    def verify(*args, **kwargs):
        calls.append(kwargs["names"])
        return actual(*args, **kwargs)

    monkeypatch.setattr(module, "verify_snapshot", verify)
    assert checks.verify(["sample"])["newly_hashed_files"] == 1
    assert checks.verify(["sample"])["newly_hashed_files"] == 0
    assert calls == [["sample"]]
    (tmp_path/"sample").write_bytes(b"changed size!")
    with pytest.raises(ValueError, match="Previously verified"):
        checks.verify(["sample"])
    with pytest.raises(ValueError, match="corrupted"):
        module.SnapshotChecks(tmp_path).verify(["sample"])


def test_same_size_immediate_write_cannot_hide_in_coarse_filesystem_timestamp(tmp_path):
    from evaluation.datasets.cosmos_reference import SnapshotChecks
    from vlaworldeval.assets import MANIFEST, file_record
    (tmp_path/"sample").write_bytes(b"original")
    (tmp_path/MANIFEST).write_text(json.dumps({"schema_version": 1, "repository": "test/data",
        "revision": "a"*40, "files": {"sample": file_record(tmp_path/"sample")}}))
    checks = SnapshotChecks(tmp_path)
    checks.verify(["sample"])
    (tmp_path/"sample").write_bytes(b"changed!")
    with pytest.raises(ValueError):
        checks.verify(["sample"])


def test_full_holdout_cli_cannot_silently_select_a_smaller_window(tmp_path):
    from scripts.benchmarks.native_replay import main
    with pytest.raises(ValueError, match="freezes"):
        main(["--model", "cosmos20k", "--full-cosmos-holdout", "--dataset", str(tmp_path),
              "--config", "unused", "--output", str(tmp_path/"new"), "--selection", "unused",
              "--chunks", "2"])


def test_missing_instruction_is_not_synthesized_for_language_free_dynamics():
    from evaluation.datasets.droid import episode_instruction
    with pytest.raises(ValueError, match="instruction is missing"):
        episode_instruction([])
    assert episode_instruction([], required=False) == ""
    assert episode_instruction(["  move cup | secondary annotation"], required=False) == "move cup"
    with pytest.raises(ValueError, match="explicit boolean"):
        episode_instruction([], required="false")
