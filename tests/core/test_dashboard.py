import json

import pytest

from evaluation.reporting.dashboard import build_dashboard
from tests.support.builders import replay_fixture

pytestmark = pytest.mark.process


def test_report_copies_synchronized_views_and_keeps_missing_slots(tmp_path, monkeypatch):
    module, config, packet, _ = replay_fixture(tmp_path, monkeypatch)
    native_dir = tmp_path / 'native'
    module.run_native_replay(config, packet, native_dir)
    branch = tmp_path / 'branch.json'
    branch.write_text(json.dumps({'kind': 'action_branch_evaluation', 'jobs': [
        {'id': '<img src=x onerror=alert(1)>', 'status': 'completed', 'summary': 'native/summary.json'},
        {'id': 'missing', 'status': 'not_run'}], 'branch_contrasts': []}))
    result = build_dashboard(tmp_path, [branch], tmp_path / 'report')
    assert result['counts'] == {'planned': 2, 'completed': 1, 'failed': 0, 'other_or_not_run': 1,
                                'with_reference_metrics': 0, 'reference_unknown': 2}
    text = (tmp_path / 'report/index.html').read_text()
    assert '<img src=x' not in text and '&lt;img' in text
    assert 'Play views together' in text and 'Timeline' in text
    assert len(list((tmp_path / 'report/media').glob('*.mp4'))) == 2
    data = json.loads((tmp_path / 'report/report.json').read_text())
    assert all(v['file'].startswith('media/') for v in data['cases'][0]['videos'])
    with pytest.raises(FileExistsError):
        build_dashboard(tmp_path, [branch], tmp_path / 'report')


def test_mutated_native_video_is_not_reported_as_completed(tmp_path, monkeypatch):
    module, config, packet, _ = replay_fixture(tmp_path, monkeypatch)
    output = tmp_path / 'native'
    module.run_native_replay(config, packet, output)
    (output / 'front.mp4').write_bytes(b'changed')
    r = build_dashboard(tmp_path, [output / 'summary.json'], tmp_path / 'report')
    assert r['counts']['failed'] == 1 and r['counts']['completed'] == 0
    assert not list((tmp_path / 'report/media').iterdir())


def test_report_does_not_silently_accept_unknown_source_format(tmp_path):
    source = tmp_path / 'unrecognized.json'
    source.write_text('{}')
    with pytest.raises(ValueError, match='Expected'):
        build_dashboard(tmp_path, [source], tmp_path / 'report')
    assert not (tmp_path / 'report').exists()


def test_planned_episode_index_matches_executed_episode(tmp_path):
    from evaluation.reporting.dashboard import _cases
    source = tmp_path / 'evaluation.json'
    source.write_text(json.dumps({'preflight': {'selection': [
        {'id': 'loop', 'episode_index': 2, 'seed': 0},
        {'id': 'loop', 'episode_index': 3, 'seed': 0}]},
        'execution': {'jobs': [{'id': 'loop', 'episode': 2, 'seed': 0,
                               'status': 'completed', 'output': 'run'}]}}))
    cases, _ = _cases(tmp_path, source)
    assert [c['status'] for c in cases] == ['completed', 'not_run']
    assert cases[0]['directory'] == tmp_path / 'run'
