"""Public manuals, presentation assets and configured pairs stay in sync."""
import hashlib
import json
from pathlib import Path
import re

import pytest

from evaluation.validation.documentation import validate_documentation
from scripts.acceptance.export_source import PRESENTATION_ASSETS, allowed, public_file_size_limit


ROOT = Path(__file__).resolve().parents[1]


def test_language_manuals_have_matching_pages_and_no_mixed_root_notes():
    english = {p.name for p in (ROOT / "docs/en").glob("*.md")}
    chinese = {p.name for p in (ROOT / "docs/zh").glob("*.md")}
    assert english == chinese and len(english) == 12
    assert {p.name for p in (ROOT / "docs").glob("*.md")} == {"README.md"}
    for language, other in (("en", "zh"), ("zh", "en")):
        for name in english:
            assert f"../{other}/{name}" in (ROOT / "docs" / language / name).read_text()


def test_readme_pair_table_covers_every_registered_pair_once():
    registry = json.loads((ROOT / "configs/runtime_registry.json").read_text())
    expected = {(vla, pair["world_model"]) for pair in registry["pairs"] for vla in pair["vlas"]}
    for language in ("en", "zh"):
        guide = (ROOT / "docs" / language / "compatibility.md").read_text()
        suites = re.findall(r"\]\(../../(configs/experiments/[^)]+)\)", guide)
        actual = []
        for suite in suites:
            config = json.loads((ROOT / suite).read_text())
            manifest = json.loads((ROOT / config["pair_manifest"]).read_text())
            actual.append((manifest["vla"]["adapter"], manifest["world_model"]["adapter"]))
        assert len(actual) == len(set(actual)) == len(expected) == 16
        assert set(actual) == expected
    for name in ("README.md", "README_CN.md"):
        text = (ROOT / name).read_text()
        assert "16" in text and "docs/assets/framework.svg" in text
        assert "state provider" not in text.lower()
        assert "statebridge" not in text.lower() and "pilworld" not in text.lower()


def test_reviewed_media_match_provenance_and_export_limits():
    record = json.loads((ROOT / "docs/assets/provenance.json").read_text())
    assert record["demo"]["task_success_verified"] is False
    for asset in record["assets"]:
        path = ROOT / asset["path"]
        assert asset["path"] in PRESENTATION_ASSETS
        assert 0 < path.stat().st_size <= public_file_size_limit(Path(asset["path"]))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == asset["sha256"]
        assert allowed(Path(asset["path"]))
    assert not allowed(Path("docs/assets/demos/other-result.mp4"))
    assert not allowed(Path("results/rollout.mp4"))


def test_readme_demonstrations_distinguish_replay_from_policy_success():
    record = json.loads((ROOT / "docs/assets/provenance.json").read_text())
    replay = record["featured_demo"]
    assert replay["vla_calls"] == 0 and replay["world_model_calls"] == 60
    assert replay["source_frames"] == 961 and replay["duration_seconds"] > 64
    long = record["long_demo"]
    assert long["vla_calls"] == 0 and long["world_model_calls"] == 130
    assert long["source_frames"] == 2081 and long["duration_seconds"] > 138
    assert not replay["task_success_verified"]
    assert record["demo"]["duration_seconds"] > 20
    assert replay["prediction_side"] == "left" and replay["recorded_reference_side"] == "right"
    for name, heading in (("README.md", "## Acknowledgements"), ("README_CN.md", "## 致谢")):
        text = (ROOT / name).read_text()
        assert "cosmos-droid-task.mp4" in text and "64.1" in text
        assert "cosmos-droid-long.mp4" in text and "138.7" in text
        assert '<img src="docs/assets/framework.png"' in text
        assert 'href="docs/assets/framework.svg"' in text
        assert "awesome-readme" not in text and heading in text
        assert "https://github.com/world-model-eval/world-model-eval" in text
        assert "https://github.com/NVIDIA/Isaac-GR00T" in text


def test_overview_diagram_is_concise_and_has_reference_isolation():
    from xml.etree import ElementTree
    diagram = ElementTree.parse(ROOT / "docs/assets/framework.svg")
    texts = ["".join(node.itertext()) for node in diagram.iter()
             if node.tag.endswith("}text")]
    assert len(" ".join(texts).split()) < 50
    assert "Recorded actions" in texts and "Real reference" in texts


def test_larger_media_budgets_are_explicit_not_a_general_export_bypass():
    assert public_file_size_limit(Path("docs/assets/demos/cosmos-droid-task.mp4")) == 16*1024**2
    assert public_file_size_limit(Path("docs/assets/demos/cosmos-droid-long.mp4")) == 16*1024**2
    assert public_file_size_limit(Path("docs/assets/demos/cosmos-droid-task.gif")) == 8*1024**2
    assert public_file_size_limit(Path("docs/assets/other.mp4")) == 2*1024**2
    assert not allowed(Path("docs/assets/other.mp4"))


def test_readme_numeric_table_matches_compact_reference():
    reference = json.loads((ROOT / "docs/assets/reference-results.json").read_text())
    assert len(reference["records"]) == 6
    for name in ("README.md", "README_CN.md"):
        text = (ROOT / name).read_text()
        for record in reference["records"]:
            for key in ("ssim_group_mean_range", "lpips_group_mean_range"):
                low, high = record[key]
                display = f"{low:.3f}" if low == high else f"{low:.3f}–{high:.3f}"
                assert display in text


def test_documentation_checker_includes_html_images_and_video_links(tmp_path):
    for name in ("README.md", "README_CN.md", "CONTRIBUTING.md", "LICENSE_SCOPE.md",
                 "adapters/README.md", "cli/README.md", "evaluation/README.md",
                 "scripts/README.md", "configs/README.md"):
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("Documentation")
    (tmp_path / "README.md").write_text('<img src="docs/missing.png">')
    with pytest.raises(ValueError, match="docs/missing.png"):
        validate_documentation(tmp_path)
    (tmp_path / "README.md").write_text('<a href="docs/missing.mp4">Video</a>')
    with pytest.raises(ValueError, match="docs/missing.mp4"):
        validate_documentation(tmp_path)
