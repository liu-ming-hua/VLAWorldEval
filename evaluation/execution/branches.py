"""Hash-bound action branches through the existing native replay executor."""
from itertools import combinations
import json
from pathlib import Path
import re
import time

import numpy as np

from evaluation.execution.transport import read_packet
from evaluation.manifests import ActionSpec, VideoSpec
from evaluation.planning.registry import project_path
from evaluation.reporting.support import digest
from evaluation.validation.parity import compare_values
from vlaworldeval.atomic_json import replace_owned_json


def asset(root, record):
    if not isinstance(record, dict) or set(record) != {'path', 'sha256'}:
        raise ValueError('Assets require relative path and SHA256')
    if Path(record['path']).is_absolute():
        raise ValueError('Branch asset paths must be workspace-relative')
    path = project_path(root, record['path'])
    if not path.is_file() or digest(path) != record['sha256']:
        raise ValueError(f'Missing or changed branch asset: {record["path"]}')
    return path


def validate_branches(root, manifest):
    root, manifest = Path(root).resolve(), Path(manifest)
    value = json.loads(manifest.read_text())
    if value.get('schema_version') != 1 or not value.get('families'):
        raise ValueError('Nonempty version-1 branch manifest required')
    seeds = value['seeds']
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)):
        raise ValueError('Distinct uint32 seeds required')
    config = json.loads(asset(root, value['config']).read_text())
    if config['schema_version'] != 1:
        raise ValueError('Branch v1 supports numeric native replay protocols only')
    spec, video = ActionSpec.from_dict(config['action']), VideoSpec.from_dict(config['video'])
    jobs, seen = [], set()
    for family in value['families']:
        fid = family['id']
        if not isinstance(fid, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', fid) or fid in seen:
            raise ValueError('Unique safe family IDs required')
        seen.add(fid)
        condition = family['initial_condition']
        if condition['kind'] not in {'observation_only', 'simulator_snapshot'}:
            raise ValueError('Unknown initial-condition claim')
        if condition['kind'] == 'simulator_snapshot':
            asset(root, condition['evidence'])
        branches = family['branches']
        if len(branches) < 2:
            raise ValueError('Each family needs at least two distinct action branches')
        base = None
        reference_indices = None
        branch_ids, actions_seen = set(), []
        for branch in branches:
            bid = branch['id']
            if not isinstance(bid, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', bid) or bid in branch_ids:
                raise ValueError('Unique safe branch IDs required')
            branch_ids.add(bid)
            packet_path = asset(root, branch['input'])
            packet = read_packet(packet_path)
            if packet['protocol'] != config['action']:
                raise ValueError('Branch action protocol differs from replay config')
            actions = np.asarray(packet['actions'])
            if actions.ndim != 3 or len(actions) == 0:
                raise ValueError('Branch actions must be [chunks,horizon,dimension]')
            for chunk in actions:
                spec.wrap(chunk, source='branch_manifest')
            context = {'initial_views': packet['initial_views'], 'reset_args': packet['reset_args'],
                       'instruction': packet['source'].get('instruction')}
            if base is None:
                base = context
                shape = actions.shape
            if compare_values(base, context)[0] or actions.shape != shape:
                raise ValueError('Branches must share complete reset/history/views/instruction and action timeline')
            if any(np.array_equal(actions, previous) for previous in actions_seen):
                raise ValueError('Duplicate action branch is not an intervention')
            actions_seen.append(actions)
            reference = asset(root, branch['reference']) if branch.get('reference') else None
            if reference:
                from evaluation.reporting.replay_metrics import validate_reference
                from evaluation.contracts import as_rgb_history
                initial = {k: as_rgb_history(v, name=k)[-1:] for k, v in packet['initial_views'].items()}
                truth = read_packet(reference)
                validate_reference(truth, config, packet, initial, input_sha256=digest(packet_path))
                if reference_indices is not None and compare_values(reference_indices, truth['frame_indices'])[0]:
                    raise ValueError('Branch reference timelines differ')
                reference_indices = truth['frame_indices']
            for seed in seeds:
                jobs.append({'id': f'{fid}__{bid}__seed{seed}', 'family': fid, 'branch': bid,
                             'seed': seed, 'input': branch['input'], 'reference': branch.get('reference'),
                             'initial_condition_kind': condition['kind'], 'status': 'not_run',
                             'planned_chunks': len(actions), 'expected_frames': video.frame_count(len(actions))})
    return value, config, jobs


def run_branches(root, manifest, output, *, execute=False):
    from evaluation.execution.native_replay import run_native_replay
    root = Path(root).resolve()
    manifest, output = project_path(root, str(manifest)), project_path(root, str(output))
    value, config, jobs = validate_branches(root, manifest)
    frozen = digest(manifest)
    if not execute:
        return {'status': 'validated_not_executed', 'planned': len(jobs), 'families': len(value['families']),
                'with_reference': sum(bool(j['reference']) for j in jobs), 'execution_performed': False}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'manifest.json').write_bytes(manifest.read_bytes())
    report = {'schema_version': 1, 'kind': 'action_branch_evaluation', 'status': 'running',
              'manifest_sha256': frozen, 'planned': len(jobs), 'independent_families': len(value['families']),
              'execution_performed': True, 'jobs': jobs, 'branch_contrasts': [],
              'physical_reliability_verified': False,
              'scope': 'Identical observed/reset context; snapshot provenance supplied by caller, not restoration certification. Seeds are not independent physical executions.'}
    journal = output / 'summary.json'
    journal.write_text(json.dumps(report, indent=2))
    try:
        for job in jobs:
            if digest(manifest) != frozen:
                raise ValueError('Branch manifest changed during execution')
            asset(root, value['config'])
            packet = asset(root, job['input'])
            reference = asset(root, job['reference']) if job['reference'] else None
            configs = output / 'configs'
            configs.mkdir(exist_ok=True)
            cfg = configs / (job['id'] + '.json')
            cfg.write_text(json.dumps({**config, 'seed': job['seed']}, indent=2))
            target = output / 'runs' / job['id']
            job['summary'] = str((target / 'summary.json').relative_to(root))
            started = time.monotonic()
            try:
                result = run_native_replay(cfg, packet, target, reference_path=reference)
                job['status'] = result['status']
            except Exception as exc:
                job.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            finally:
                job['seconds_including_startup'] = time.monotonic() - started
                replace_owned_json(journal, report)
        report['branch_contrasts'] = branch_contrasts(root, jobs)
        report['status'] = 'completed' if all(j['status'] == 'completed' for j in jobs) else 'completed_with_failures'
    except BaseException:
        report['status'] = 'interrupted_or_invalidated'
        raise
    finally:
        report['counts'] = {s: sum(j['status'] == s for j in jobs) for s in ('completed', 'failed', 'not_run')}
        report['counts']['reference_unknown'] = sum(not j['reference'] for j in jobs)
        replace_owned_json(journal, report)
    return report


def branch_contrasts(root, jobs):
    """Decoded-video contrast error; explicitly not a contact/state metric."""
    import mediapy as media
    results = []
    for left, right in combinations(jobs, 2):
        if left['family'] != right['family'] or left['seed'] != right['seed']:
            continue
        row = {'left': left['id'], 'right': right['id'], 'status': 'unknown', 'views': {},
               'randomness_matching': 'same_seed_only_not_verified_noise_trace'}
        results.append(row)
        if any(j['status'] != 'completed' or not j['reference'] for j in (left, right)):
            row['reason'] = 'Both completed predictions and both executed references required'
            continue
        try:
            refs = [read_packet(asset(root, j['reference'])) for j in (left, right)]
            for view in refs[0]['views']:
                videos = [np.asarray(media.read_video(project_path(root, j['summary']).parent / f'{view}.mp4')) for j in (left, right)]
                expected = [np.asarray(r['views'][view]) for r in refs]
                if any(v.shape != expected[0].shape for v in videos + expected):
                    raise ValueError('Branch prediction/reference geometry or horizon differs')
                error = ((videos[0][1:].astype(float) - videos[1][1:].astype(float))
                         - (expected[0][1:].astype(float) - expected[1][1:].astype(float))) / 255
                row['views'][view] = {'paired_response_mse_01': float(np.mean(error**2)),
                                      'frames': len(error)}
            row.update(status='scored', metric_domain='decoded_RGB_prediction_vs_raw_RGB_reference',
                       physical_action_tracking_verified=False)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            row.update(reason=str(exc), views={})
    return results
