import pytest

from adapters.world_models.cosmos3_nano_droid_fd import context_parallel_degree
from adapters.backends.cosmos import local_tokenizer_config


def test_old_text_tokenizer_schema_remains_compatible(tmp_path):
    original = {"_target": "create_qwen2_tokenizer_with_download", "pretrained_model_name": "remote"}
    value = local_tokenizer_config(original, tmp_path / "model", tmp_path / "text")
    assert value["pretrained_model_name"] == str(tmp_path / "text")
    assert "tokenizer_type" not in value
    assert original["pretrained_model_name"] == "remote"


def test_policy_multimodal_processor_uses_checkpoint_local_assets(tmp_path):
    for name in ("preprocessor_config.json", "video_preprocessor_config.json", "tokenizer.json"):
        (tmp_path / name).write_text("{}")
    original = {"_target": "build_processor_lazy", "tokenizer_type": "remote", "repository": "remote",
                "revision": "remote", "pretrained_model_name": "incorrect keyword"}
    value = local_tokenizer_config(original, tmp_path, tmp_path / "text")
    assert value["tokenizer_type"] == str(tmp_path)
    assert not ({"repository", "revision", "pretrained_model_name"} & value.keys())
    assert original["tokenizer_type"] == "remote"


def test_processor_requires_local_assets_and_known_factory(tmp_path):
    with pytest.raises(FileNotFoundError, match="processor asset"):
        local_tokenizer_config({"_target": "build_processor_lazy"}, tmp_path, tmp_path)
    with pytest.raises(ValueError, match="factory"):
        local_tokenizer_config({"_target": "unknown"}, tmp_path, tmp_path)


@pytest.mark.parametrize("world_size,expected", [(1,1),(2,2),(3,1),(4,4),(6,2),(8,8)])
def test_context_parallel_is_independent_of_weight_shards(world_size, expected):
    assert context_parallel_degree(world_size, 32, 8) == expected


def test_bad_attention_contract_rejected():
    with pytest.raises(ValueError):
        context_parallel_degree(0, 32, 8)
    with pytest.raises(ValueError):
        context_parallel_degree(3, 32, 7)
