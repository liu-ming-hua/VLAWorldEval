"""Protocol comparisons exclude operational paths, not numerical conditions."""
from evaluation.validation.cosmos_parity import canonical_sample


def test_cosmos_canonicalization_keeps_numerics_and_conditioning_protocol():
    sample = {'name': 'call', 'output_dir': '/run/native', 'vision_path': '/run/native/a.png',
              'action_path': '/run/native/a.json', 'fps': 15, 'num_frames': 17,
              'model_mode': 'forward_dynamics', 'seed': 2, 'nested': {'shift': 10., 'guidance': 1., 'name': 'sampler'}}
    actual = canonical_sample(sample)
    assert actual == {'fps': 15, 'num_frames': 17, 'model_mode': 'forward_dynamics',
                      'seed': 2, 'nested': {'shift': 10., 'guidance': 1., 'name': 'sampler'}}
    assert sample['name'] == 'call'
    assert canonical_sample({**sample, 'fps': 5}) != actual


def test_acceptance_preparation_rejects_unknown_model_without_creating_output(tmp_path):
    import pytest
    from scripts.acceptance.prepare_world_model_cases import prepare
    with pytest.raises(ValueError, match='Unsupported'):
        prepare(tmp_path, 'unknown', tmp_path/'output')
    assert not (tmp_path/'output').exists()
