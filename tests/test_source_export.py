"""Public source exports preserve framework/CI and reject runtime or secret payloads."""
import io
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from scripts.acceptance.export_source import ROOT_FILES, export_source


def source_archive(tmp_path, extra=()):
    path = tmp_path/'source.tar.gz'
    required = ROOT_FILES | {'.github/workflows/quality.yml', 'scripts/setup/core.sh', 'environment/registry.toml'}
    with tarfile.open(path, 'w:gz') as stream:
        for name, data in [*((n, b'fixture\n') for n in sorted(required)), *extra]:
            info = tarfile.TarInfo('vlaworldeval-0.1.0/'+name)
            info.size = len(data)
            stream.addfile(info, io.BytesIO(data))
    return path


def test_export_is_new_clean_source_with_ci_and_no_history(tmp_path):
    archive = source_archive(tmp_path, [('vlaworldeval.egg-info/PKG-INFO', b'metadata')])
    out = tmp_path/'public'
    result = export_source(archive, out)
    assert result['status'] == 'exported_not_published'
    assert (out/'.github/workflows/quality.yml').is_file()
    assert not (out/'.git').exists() and not (out/'vlaworldeval.egg-info').exists()
    assert (out/'release_manifest.json').is_file()
    with pytest.raises(FileExistsError):
        export_source(archive, out)


@pytest.mark.parametrize('name,data', [
    ('results/config.json', b'{}'), ('third_party/upstream.py', b'print(1)'),
    ('.git/config', b'remote'), ('checkpoints/model.safetensors', b'weights'),
    ('environment/core/bin/python', b'binary'), ('../escape.py', b'bad'),
    ('README.md', b'duplicate'), ('docs/key.md', b'hf_'+b'x'*32),
    ('configs/private.key', b'private'),
    ('docs/assets/demos/unreviewed.mp4', b'video'),
])
def test_export_rejects_payload_before_creating_output(tmp_path, name, data):
    with pytest.raises(ValueError):
        export_source(source_archive(tmp_path, [(name, data)]), tmp_path/'public')
    assert not (tmp_path/'public').exists()


def test_export_rejects_symlinks(tmp_path):
    archive = source_archive(tmp_path)
    with tarfile.open(archive, 'r:gz') as stream:
        members = [(m, stream.extractfile(m).read()) for m in stream]
    with tarfile.open(archive, 'w:gz') as stream:
        for member, data in members:
            stream.addfile(member, io.BytesIO(data))
        link = tarfile.TarInfo('vlaworldeval-0.1.0/docs/link')
        link.type, link.linkname = tarfile.SYMTYPE, '/outside'
        stream.addfile(link)
    with pytest.raises(ValueError):
        export_source(archive, tmp_path/'public')


def test_export_checks_reviewed_presentation_media_hashes(tmp_path):
    name = 'docs/assets/demos/groot-cosmos.mp4'
    data = b'reviewed-demo'
    provenance = json.dumps({'assets': [{'path': name,
        'sha256': hashlib.sha256(data).hexdigest()}]}).encode()
    valid = source_archive(tmp_path, [(name, data), ('docs/assets/provenance.json', provenance)])
    assert export_source(valid, tmp_path/'valid')['status'] == 'exported_not_published'
    invalid = source_archive(tmp_path, [(name, b'changed'), ('docs/assets/provenance.json', provenance)])
    with pytest.raises(ValueError, match='checksum mismatch'):
        export_source(invalid, tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()


def test_git_ignores_all_result_formats_and_upstream_clones():
    root = Path(__file__).resolve().parents[1]
    paths = ['results/example/report.json', 'results/example/condition.png',
             'third_party/ewmbench/model.py', 'third_party/co-tracker/.git/config']
    run = subprocess.run(['git', 'check-ignore', '--no-index', *paths], cwd=root,
                         capture_output=True, text=True)
    # Source-distribution validation may run outside a Git checkout.
    if run.returncode == 128:
        pytest.skip('Git ignore behavior is checked in a checkout, not a source distribution')
    assert run.returncode == 0 and set(run.stdout.splitlines()) == set(paths)
