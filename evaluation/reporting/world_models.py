"""Read-only world-model route inventory; registration is never execution evidence."""
from pathlib import Path
import json

from evaluation.planning.catalog import load_model_catalog
from evaluation.planning.registry import load_runtime_registry
from evaluation.reporting.support import support_report


def verification_status(records, pairs=()):
    """Report independent evidence axes, never infer quality from route existence."""
    def status(field, accepted):
        matching = [r for r in records if r.get(field, {}).get('status') in accepted]
        if not matching:
            return 'not_recorded'
        return ('historical_pass' if any(r.get('current_adapter_sources_match') is True for r in matching)
                else 'historical_pass_source_changed')
    replay = status('long_replay', {'completed'})
    parity = status('native_entry_parity', {'passed'})
    real = replay if replay != 'not_recorded' else parity
    if real == 'not_recorded' and any(p.get('real_weight_loop', {}).get('status') in {'passed', 'historical_pass', 'completed'} for p in pairs):
        real = 'historical_pair_run_only'
    return {'interface_tests': status('interface_tests', {'passed'}),
            'real_weight_execution': real, 'native_replay': replay,
            'official_entry_parity': parity,
            'ground_truth_comparison': status('reference_comparison', {'passed', 'completed'}),
            'physical_reliability': 'not_verified'}


def render_verification_table(report):
    """Compact user-facing matrix; the JSON report retains evidence and blockers."""
    columns = [('interface_tests', 'Interface tests'), ('real_weight_execution', 'Real weights'),
               ('native_replay', 'Native replay'), ('official_entry_parity', 'Entry parity'),
               ('ground_truth_comparison', 'GT comparison')]
    lines = ['| Checkpoint | ' + ' | '.join(label for _, label in columns) + ' |',
             '| --- | ' + ' | '.join('---' for _ in columns) + ' |']
    for row in report['world_models']:
        lines.append('| ' + row['model'] + ' | ' + ' | '.join(row['verification'][key] for key, _ in columns) + ' |')
    return '\n'.join(lines) + '\n\nHistorical evidence only. not_recorded means no published receipt, not model failure. No physical-reliability certification.'


def world_model_report(root, model=None):
    root = Path(root)
    catalog = load_model_catalog(root)['world_models']
    if model is not None and model not in catalog:
        raise ValueError(f"Unknown world model {model!r}; choose from: {', '.join(catalog)}")
    registry = load_runtime_registry(root)
    support = support_report(root)
    configs = []
    for path in sorted((root / 'configs/replay').glob('*.json')):
        configs.append((path.relative_to(root).as_posix(), json.loads(path.read_text())))
    rows = []
    for key, entry in catalog.items():
        if model is not None and key != model:
            continue
        adapter = entry['adapter']
        runtime = registry.models['world_models'].get(adapter, {})
        # Explicit wrapper alias: a shared environment alone does not identify weights.
        native_adapter = {
            "cosmos3_nano_droid_fd_20k_ema": "Cosmos3DroidFDNativeAdapter",
        }.get(key, adapter)
        routes = [(path, cfg) for path, cfg in configs
                  if cfg.get("adapter") == native_adapter
                  and cfg.get("environment") == entry["environment"]]
        pairs = [p for p in support['pairs'] if p['world_model'] == adapter]
        evidence = [r for r in support['runtime_acceptance']
                    if r.get('model_key') == key]
        blockers = []
        if not routes:
            blockers.append('No shared native replay configuration; adapter registration alone is insufficient.')
        if not pairs:
            blockers.append('No registered VLA closed-loop pair for this checkpoint profile.')
        if not evidence:
            blockers.append('No compact native parity/long-replay acceptance receipt in the published catalog.')
        protocol_gaps = {
            'a2world': 'Bundled DROID example has one 20x14 chunk and no recorded future RGB; requires gated NVIDIA tokenizer.',
            'ge_sim_v2': 'Requires an external GE-Sim server and calibrated episode bundle; episode-conditioned replay is not an action-only counterfactual test.',
            'tau0_wm_simulator': 'Local TacoPlay sample is 7-D; released simulator API requires 20-D relative EEF actions (26-D after server controls). No matching TacoPlay checkpoint was found in the checked official model tree. The default API returns terminal views; an opt-in transport patch exposes full future video, unit-tested but not native-data validated. Zero-action smoke is not native data validation.',
            'gigaworld1': 'Released route consumes control video, not raw robot actions; no calibrated action-to-control conversion is registered.',
            'cosmos3_nano': 'Base Diffusers checkpoint has domain-specific action protocols; DROID post-trained replay configuration cannot be reused as validation of base weights.',
        }
        if key in protocol_gaps:
            blockers.append(protocol_gaps[key])
        replay = []
        for path, cfg in routes:
            command = ('python -m cli replay --config ' + path
                       + ' --input CASE/input.npz'
                       + ('' if key in {'a2world', 'ge_sim_v2'} else ' --reference CASE/reference.npz')
                       + ' --output results/MY_NEW_RUN')
            replay.append({'config': path, 'adapter': cfg['adapter'],
                           'environment': cfg['environment'],
                           'reference_available_in_example': False if key in {'a2world', 'ge_sim_v2'} else None,
                           'prepare_template': ('python -m cli prepare-replay --config ' + path
                               + ' --dataset datasets/a2world/official/droid --output CASE/input.npz --chunks 1'
                               if key == 'a2world' else None),
                           'action': cfg.get('action'), 'video': cfg.get('video'),
                           'admission_template': command + ' --admit-only',
                           'execution_template': command})
        rows.append({'model': key, 'adapter': adapter, 'environment': entry['environment'],
                     'declared_status': entry.get('status'),
                     'declared_integration': entry.get('integration'),
                     'input': entry.get('input'), 'output': entry.get('output'),
                     'training_domain': entry.get('training_domain'),
                     'declared_assets': runtime.get('assets', {}),
                     'native_replay_routes': replay,
                     'native_acceptance_evidence': evidence,
                     'verification': verification_status(evidence, pairs),
                     'closed_loop_evidence': pairs, 'gaps': blockers})
    return {'schema_version': 1, 'world_models': rows, 'execution_performed': False,
            'scope': 'Declared routes plus published historical evidence. No environment, weight, '
                     'GPU or task-success certification. Configurations are alternatives, not rankings.',
            'instructions': [
                'Use the installation and model manuals for isolated setup and checkpoint downloads.',
                'Prepare a protocol-matched CASE input; use a separate bound reference only when real future RGB is available; '
                'CASE and MY_NEW_RUN in commands are placeholders, not bundled datasets.',
                'Choose the intended version explicitly; old replay configurations remain visible.',
                'Run admission first, then execution with a fresh output directory.',
                'Build a portable report: python -m cli report '
                '--source results/MY_NEW_RUN/summary.json --output results/MY_NEW_REPORT',
                'Reference future images are scored after generation and never passed to the model.',
            ]}
