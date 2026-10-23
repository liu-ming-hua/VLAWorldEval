"""Export reviewed source and allowlisted presentation media, without runtime assets."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile


ROOT_FILES = {"pyproject.toml", "MANIFEST.in", ".gitignore", "README.md", "README_CN.md",
              "CONTRIBUTING.md", "LICENSE", "LICENSE_SCOPE.md"}
CODE_DIRS = {"adapters", "cli", "evaluation", "vlaworldeval", "scripts", "tests", "configs", "docs"}
TEXT_SUFFIXES = {".py", ".sh", ".md", ".patch", ".json", ".yaml", ".yml", ".toml", ".txt", ".in"}
PRESENTATION_ASSETS = {
    'docs/assets/banner.png', 'docs/assets/framework.svg',
    'docs/assets/framework.png',
    'docs/assets/demos/groot-cosmos.gif', 'docs/assets/demos/groot-cosmos.mp4',
    'docs/assets/demos/cosmos-droid-task.gif', 'docs/assets/demos/cosmos-droid-task.mp4',
    'docs/assets/demos/cosmos-droid-long.png', 'docs/assets/demos/cosmos-droid-long.mp4',
    'docs/assets/provenance.json', 'docs/assets/reference-results.json',
}
SECRET = re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
                    rb"\bhf_[A-Za-z0-9]{30,}\b|\bAKIA[A-Z0-9]{16}\b|\bsk-[A-Za-z0-9_-]{32,}\b")


def public_file_size_limit(path):
    """Only named, checksum-bound long videos get a larger per-file budget."""
    limits = {'docs/assets/demos/cosmos-droid-task.mp4': 16,
              'docs/assets/demos/cosmos-droid-long.mp4': 16,
              'docs/assets/demos/cosmos-droid-task.gif': 8}
    return limits.get(str(path), 2) * 1024**2


def allowed(path):
    if any(p.startswith('.') and p not in {'.github', '.gitignore'} for p in path.parts):
        return False
    if len(path.parts) == 1:
        return path.name in ROOT_FILES
    top = path.parts[0]
    if path.parts[:2] == ('docs', 'assets'):
        return str(path) in PRESENTATION_ASSETS
    if top in CODE_DIRS:
        return path.suffix in TEXT_SUFFIXES
    if top == '.github':
        return len(path.parts) == 3 and path.parts[1] == 'workflows' and path.suffix in {'.yml', '.yaml'}
    if top == 'environment':
        return (str(path) == 'environment/registry.toml' or
                len(path.parts) == 3 and path.parts[1] in {'locks', 'requirements'} and path.suffix in {'.txt', '.in'})
    if top in {'datasets', 'checkpoints'}:
        return path.name == 'README.md' or top == 'datasets' and path.name == 'manifest.json'
    return False


def export_source(sdist, output):
    sdist, output = Path(sdist).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('Choose a new export directory; existing files are never overwritten')
    files, prefix, total = {}, None, 0
    with tarfile.open(sdist, 'r:gz') as archive:
        for index, member in enumerate(archive):
            name = PurePosixPath(member.name)
            if (index >= 5000 or name.is_absolute() or '..' in name.parts or not name.parts
                    or not (member.isdir() or member.isfile())):
                raise ValueError('Unsafe source archive member: ' + member.name)
            if prefix is None:
                prefix = name.parts[0]
            if name.parts[0] != prefix:
                raise ValueError('Source archive must have one package root')
            if member.isdir():
                continue
            path = PurePosixPath(*name.parts[1:])
            if path.name in {'PKG-INFO', 'setup.cfg'} and len(path.parts) == 1:
                continue
            if path.parts and path.parts[0] == 'vlaworldeval.egg-info':
                continue  # Generated build metadata is not a source contribution.
            if not allowed(path) or path in files or not 0 <= member.size <= public_file_size_limit(path):
                raise ValueError('Unexpected or duplicate public source file: ' + str(path))
            total += member.size
            if total > 64*1024**2:
                raise ValueError('Source exceeds the bounded export size')
            data = archive.extractfile(member).read()
            if SECRET.search(data):
                raise ValueError('Credential-like content requires review: ' + str(path))
            files[path] = (data, bool(member.mode & 0o111))
    required = ROOT_FILES | {'.github/workflows/quality.yml', 'scripts/setup/core.sh', 'environment/registry.toml'}
    if not required.issubset(map(str, files)):
        raise ValueError('Source archive lacks required release files')
    media = {str(p): data for p, (data, _) in files.items()
             if str(p) in PRESENTATION_ASSETS and p.suffix not in {'.json'}}
    if media:
        try:
            provenance = json.loads(files[PurePosixPath('docs/assets/provenance.json')][0])
            declared = {item['path']: item['sha256'] for item in provenance['assets']}
            if any(declared.get(p) != hashlib.sha256(data).hexdigest() for p, data in media.items()):
                raise ValueError('Presentation asset checksum mismatch')
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError('Presentation assets require a valid provenance manifest') from exc
    output.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for path, (data, executable) in files.items():
        target = output.joinpath(*path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(0o755 if executable else 0o644)
        manifest[str(path)] = hashlib.sha256(data).hexdigest()
    record = {'schema_version': 1, 'status': 'exported_not_published', 'source_archive': sdist.name,
              'source_archive_sha256': hashlib.sha256(sdist.read_bytes()).hexdigest(),
              'files': manifest, 'git_history_included': False, 'model_assets_included': False,
              'credential_scan': 'limited_pattern_scan_not_a_security_or_legal_clearance'}
    (output/'release_manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    return {'status': record['status'], 'files': len(files), 'output': str(output)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdist', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_source(args.sdist, args.output)))
