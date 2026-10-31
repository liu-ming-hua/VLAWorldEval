"""Freeze native long-replay acceptance cases; run inference with `native-study`."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.planning.registry import project_path  # noqa: E402


def prepare(root, model, output):
    root = Path(root).resolve()
    output = project_path(root, str(output))
    if model == 'ctrl_world':
        from evaluation.datasets.ctrl_world import prepare_released_validation
        return prepare_released_validation(root, 'third_party/ctrl-world',
            'configs/replay/ctrl_world_native_v2.json', output, mode='long')
    if model != 'cosmos_droid':
        raise ValueError('Unsupported acceptance profile')
    from evaluation.datasets.cosmos_reference import prepare_native_case, SnapshotChecks
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((root/'configs/replay/cosmos_droid_native_v1.json').read_text())
    config.update(gpus=[0, 1], memory_gib=38)
    path = output/'config.json'
    path.write_text(json.dumps(config, indent=2))
    checks = SnapshotChecks(root/'datasets/cosmos3_droid')
    cases = []
    for episode in (0, 351, 795):
        case = output/f'cosmos_ep{episode}'
        prepare_native_case(root, 'datasets/cosmos3_droid/success', path, case,
                            episode=episode, chunks='all', _checks=checks)
        cases.append({'id': f'cosmos_ep{episode}', 'prepared': str(case.relative_to(root))})
    plan = {'schema_version': 1,
            'scope': 'Engineering acceptance on fixed development episodes0,351,795; all complete16-action chunks; not an independent held-out estimate',
            'cases': cases, 'unavailable': {'physical_action_truth': 'No executed alternative branches',
                'incomplete_tail': 'Excluded and counted in each packet provenance; never padded'}}
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    return {'plan': str(output/'plan.json'), 'episodes': 3, 'model_inference_performed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['ctrl_world', 'cosmos_droid'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(ROOT, args.model, args.output), indent=2))
