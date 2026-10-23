import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "relocate_runtime", Path(__file__).resolve().parents[1] / "scripts/ops/relocate_runtime.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture_root(tmp_path):
    root = tmp_path / "destination"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\n")
    env = root / "environment/core"
    (env / "bin").mkdir(parents=True)
    return root, env


def test_relocation_is_explicit_preserves_evidence_and_backups(tmp_path):
    root, env = fixture_root(tmp_path)
    old = "/data1/liuming/VLAWorldEval/environment/core"
    cfg = env / "pyvenv.cfg"
    cfg.write_text("home = /home/liuming/.local/share/uv/python/py/bin\n")
    launcher = env / "bin/demo"
    launcher.write_text(f"#!{old}/bin/python\n")
    launcher.chmod(0o755)
    evidence = root / "results/plan.json"
    evidence.parent.mkdir()
    evidence.write_text(old)
    before = cfg.read_bytes()
    assert MODULE.relocate(root, only_core=True)["changes"]
    assert cfg.read_bytes() == before
    report = MODULE.relocate(root, apply=True, only_core=True)
    assert report["changes"]
    assert str(root) in launcher.read_text()
    assert launcher.stat().st_mode & 0o777 == 0o755
    assert evidence.read_text() == old
    saved = root / ".tools/relocation/originals/environment/core/pyvenv.cfg"
    assert saved.read_bytes() == before
    assert not MODULE.relocate(root, apply=True, only_core=True)["changes"]
    assert saved.read_bytes() == before


def test_symlink_rebased_binary_untouched_external_reported(tmp_path):
    root, env = fixture_root(tmp_path)
    link = env / "bin/python"
    link.symlink_to("/home/liuming/.local/share/uv/python/py/bin/python")
    binary = env / "bin/native"
    payload = b"\x7fELF\0/data1/liuming/VLAWorldEval"
    binary.write_bytes(payload)
    external = env / "other.pth"
    external.write_text("/data0/someone/not_our_project\n")
    report = MODULE.relocate(root, apply=True, only_core=True)
    assert str(link.readlink()).startswith(str(root / ".tools/python"))
    assert binary.read_bytes() == payload
    assert any(x["reason"] == "unmapped_external_path" for x in report["unresolved"])
    assert external.read_text() == "/data0/someone/not_our_project\n"


def test_reject_active_source_and_symlink_destination(tmp_path):
    with pytest.raises(ValueError):
        MODULE.relocate(Path(MODULE.OLD_ROOT), apply=True)
    root, _ = fixture_root(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(root)
    with pytest.raises(ValueError):
        MODULE.relocate(link, apply=True)


def test_core_also_repairs_managed_python_alias_before_final_audit(tmp_path):
    root, env = fixture_root(tmp_path)
    managed = root / ".tools/python"
    (managed / "versioned/bin").mkdir(parents=True)
    (managed / "versioned/bin/python").write_text("fixture")
    (managed / "alias").symlink_to(MODULE.OLD_PYTHON + "/versioned")
    (env / "bin/python").symlink_to(MODULE.OLD_PYTHON + "/alias/bin/python")
    report = MODULE.relocate(root, apply=True, only_core=True)
    assert (env / "bin/python").exists()
    assert not report["unresolved"]


def test_repairs_project_local_uv_python_alias(tmp_path):
    root, env = fixture_root(tmp_path)
    managed = root / ".cache/uv/python"
    (managed / "versioned/bin").mkdir(parents=True)
    (managed / "versioned/bin/python").write_text("fixture")
    (managed / "alias").symlink_to(MODULE.OLD_ROOT + "/.cache/uv/python/versioned")
    (env / "bin/python").symlink_to(MODULE.OLD_ROOT + "/.cache/uv/python/alias/bin/python")
    report = MODULE.relocate(root, apply=True)
    assert (env / "bin/python").exists()
    assert not report["unresolved"]
