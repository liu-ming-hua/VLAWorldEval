import json

import pytest

from evaluation.reporting.dashboard import build_dashboard
from tests.support.builders import replay_fixture

pytestmark = pytest.mark.process


def test_demo_report_is_portable_and_rechecks_artifacts(tmp_path):
    import shutil
    from vlaworldeval.demo import run_demo

    original = tmp_path / 'original'
    run_demo(original, chunks=3)
    moved = tmp_path / 'moved'
    original.rename(moved)
    result = build_dashboard(tmp_path, [moved / 'summary.json'], tmp_path / 'report')
    assert result['counts']['completed'] == 1
    data = json.loads((tmp_path / 'report/report.json').read_text())
    case = data['cases'][0]
    assert case['synthetic'] and not case['task_success_evaluated']
    assert case['world_model_calls'] == case['vla_calls'] == 3
    assert len(case['videos']) == 2
    assert 'Synthetic engineering demo' in (tmp_path / 'report/index.html').read_text()
    (moved / 'actions.npz').unlink()
    broken = build_dashboard(tmp_path, [moved / 'summary.json'], tmp_path / 'broken')
    assert broken['counts']['completed'] == 0
    assert broken['counts']['failed'] == 1
    shutil.rmtree(moved)
    assert all((tmp_path / 'report' / v['file']).is_file() for v in case['videos'])


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


def test_native_study_report_preserves_frozen_queue_and_binds_progress(tmp_path):
    from evaluation.reporting.dashboard import _cases
    from evaluation.reporting.support import digest
    source = tmp_path / 'study.json'
    source.write_text(json.dumps({'source_plan_sha256': 'plan', 'jobs': [{'id': 'a'}, {'id': 'b'}]}))
    progress = tmp_path / 'progress.json'
    progress.write_text(json.dumps({'study_sha256': digest(source), 'jobs': {'a': {'status': 'failed', 'error': 'fixture'}}}))
    cases, _ = _cases(tmp_path, source)
    assert [c['status'] for c in cases] == ['failed', 'not_run']
    report = build_dashboard(tmp_path, [source], tmp_path / 'report')
    assert report['counts']['planned'] == 2 and report['counts']['failed'] == 1
    progress.write_text(json.dumps({'study_sha256': 'changed'}))
    with pytest.raises(ValueError, match='different frozen study'):
        _cases(tmp_path, source)
