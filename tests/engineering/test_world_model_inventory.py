"""The route index must not promote declarations to tested support."""
import json
from pathlib import Path

import pytest

from cli.parser import build_parser
from evaluation.reporting.world_models import world_model_report

ROOT = Path(__file__).resolve().parents[2]


def test_inventory_covers_catalog_and_commands_use_shared_replay():
    report = world_model_report(ROOT)
    catalog = json.loads((ROOT / 'configs/model_catalog.json').read_text())['world_models']
    assert {r['model'] for r in report['world_models']} == set(catalog)
    assert report['execution_performed'] is False
    for row in report['world_models']:
        for route in row['native_replay_routes']:
            command = route['admission_template'].split()[3:]
            args = build_parser(ROOT).parse_args(command)
            assert args.command == 'replay' and args.admit_only
            assert (args.reference is None) == (row["model"] in {"a2world", "ge_sim_v2"})
            assert (ROOT / route['config']).is_file()


def test_inventory_distinguishes_base_checkpoint_and_native_wrapper():
    base = world_model_report(ROOT, 'cosmos3_nano')['world_models'][0]
    assert not base['native_replay_routes'] and not base['closed_loop_evidence']
    tuned = world_model_report(ROOT, 'cosmos3_nano_droid_fd_20k_ema')['world_models'][0]
    assert tuned['native_replay_routes'][0]['adapter'] == 'Cosmos3DroidFDNativeAdapter'
    assert tuned['native_acceptance_evidence']
    assert base['gaps']


def test_unknown_profile_is_an_actionable_error():
    with pytest.raises(ValueError, match='Unknown world model'):
        world_model_report(ROOT, 'missing_checkpoint')


def test_evidence_axes_do_not_promote_interface_or_reference_presence():
    from evaluation.reporting.world_models import verification_status, render_verification_table

    record = {'interface_tests': {'status': 'passed'}, 'current_adapter_sources_match': True,
              'long_replay': {'status': 'failed', 'cases': [{'reference_sha256': 'abc'}]}}
    result = verification_status([record])
    assert result['interface_tests'] == 'historical_pass'
    assert result['real_weight_execution'] == 'not_recorded'
    assert result['ground_truth_comparison'] == 'not_recorded'
    record['long_replay']['status'] = 'completed'
    record['current_adapter_sources_match'] = False
    result = verification_status([record])
    assert result['native_replay'] == 'historical_pass_source_changed'
    assert result['physical_reliability'] == 'not_verified'
    table = render_verification_table({'world_models': [{'model': 'fixture', 'verification': result}]})
    assert 'not_recorded' in table and 'source_changed' in table


def test_receipts_bind_exact_checkpoint_profile():
    base = world_model_report(ROOT, 'cosmos3_nano')['world_models'][0]
    tuned = world_model_report(ROOT, 'cosmos3_nano_droid_fd_20k_ema')['world_models'][0]
    assert base['verification']['native_replay'] == 'not_recorded'
    assert tuned['verification']['native_replay'].startswith('historical_pass')
    assert build_parser(ROOT).parse_args(['world-models', '--table']).table


def test_server_only_profiles_do_not_require_a_vla_pair_to_show_evidence():
    ge = world_model_report(ROOT, 'ge_sim_v2')['world_models'][0]
    tau = world_model_report(ROOT, 'tau0_wm_simulator')['world_models'][0]
    assert ge['verification']['native_replay'] == 'historical_pass'
    assert ge['verification']['official_entry_parity'] == 'historical_pass'
    assert tau['verification']['interface_tests'] == 'historical_pass'
    assert tau['verification']['real_weight_execution'] == 'not_recorded'
