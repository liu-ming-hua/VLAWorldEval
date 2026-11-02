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
