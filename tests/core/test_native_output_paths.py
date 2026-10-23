import importlib

import pytest

from vlaworldeval.paths import fresh_output_path


def test_output_preflight_does_not_create_files_and_atomic_claim_protects_races(tmp_path):
    output = fresh_output_path(tmp_path, "results/new")
    assert not output.exists()
    output.mkdir(parents=True)
    marker = output / "user.txt"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        output.mkdir(exist_ok=False)
    with pytest.raises(FileExistsError, match="new experiment"):
        fresh_output_path(tmp_path, output)
    assert marker.read_text() == "preserve"


@pytest.mark.parametrize("kind", ["outside", "symlink", "dangling_symlink", "parent_symlink"])
def test_output_cannot_escape_or_reuse_symlinks(tmp_path, kind):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    if kind == "outside":
        value = "../outside"
    else:
        link = root / "link"
        if kind == "symlink":
            outside.mkdir()
        link.symlink_to(outside, target_is_directory=True)
        value = link / "new" if kind == "parent_symlink" else link
    with pytest.raises(ValueError):
        fresh_output_path(root, value)
    assert not (outside / "new").exists()


@pytest.mark.parametrize("module_name", ["a2world", "ge_sim_v2", "tau0_protocol", "gigaworld1"])
def test_native_cli_refuses_existing_output_before_models_or_servers(
    tmp_path, monkeypatch, module_name
):
    module = importlib.import_module(f"scripts.replay.{module_name}")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "actions_all.npy"
    marker.write_bytes(b"existing user result")
    monkeypatch.setattr("sys.argv", [module_name, "--output", str(output)])
    with pytest.raises(FileExistsError, match="new experiment"):
        module.main()
    assert marker.read_bytes() == b"existing user result"
