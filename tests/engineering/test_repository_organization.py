from pathlib import Path

import pytest

from evaluation.validation.documentation import validate_documentation


def test_current_source_links_resolve():
    report = validate_documentation(Path.cwd())
    assert report["documents"] >= 20 and report["source_links"] >= 50
    assert not report["external_links_checked"]


def test_public_cli_dispatch_is_complete_and_legacy_entries_are_retired():
    from cli.commands import COMMANDS
    from cli.parser import build_parser

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    metadata = tomllib.loads(Path("pyproject.toml").read_text())
    assert metadata["project"]["scripts"] == {"vlaworldeval": "cli.main:main"}
    for command in ("models", "environments", "capabilities", "validate", "cleanup-results"):
        args = build_parser(Path.cwd()).parse_args([command])
        assert callable(COMMANDS[args.command])


def test_model_worker_path_uses_package_root_after_module_move():
    from evaluation.execution import client

    root = Path(client.__file__).resolve().parents[2]
    assert (root / "evaluation/execution/worker.py").is_file()
    assert (root / "cli/main.py").is_file()


def test_documentation_checker_rejects_missing_source_links(tmp_path):
    (tmp_path / "README.md").write_text("[missing](missing_source.py)")
    with pytest.raises(ValueError, match="Broken source link"):
        validate_documentation(tmp_path)


def test_documentation_checker_includes_nested_model_guides_but_not_archives(tmp_path):
    for name in (
        "README.md",
        "README_CN.md",
        "CONTRIBUTING.md",
        "LICENSE_SCOPE.md",
        "adapters/README.md",
        "cli/README.md",
        "evaluation/README.md",
        "scripts/README.md",
        "configs/README.md",
    ):
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("Current documentation")
    models = tmp_path / "docs/models"
    models.mkdir(parents=True)
    guide = models / "EXAMPLE.md"
    guide.write_text("[missing](../../adapters/missing.py)")
    archive = tmp_path / "docs/archive"
    archive.mkdir()
    (archive / "OLD.md").write_text("[historical](missing_historical.py)")
    with pytest.raises(ValueError, match="docs/models/EXAMPLE.md"):
        validate_documentation(tmp_path)
    guide.write_text("[current](../../adapters/README.md)")
    report = validate_documentation(tmp_path)
    assert report["documents"] == 10
    assert report["source_links"] == 1
