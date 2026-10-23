"""Shared official Cosmos Framework loading for policy and dynamics adapters."""
from pathlib import Path
import json
import os
import sys


def local_tokenizer_config(config, model_path, text_tokenizer_path):
    """Respect the two upstream tokenizer schemas; never mix their keywords."""
    result = dict(config)
    target = str(result.get("_target", "")).rsplit(".", 1)[-1]
    if target == "create_qwen2_tokenizer_with_download":
        result["pretrained_model_name"] = str(text_tokenizer_path)
    elif target == "build_processor_lazy":
        for name in ("preprocessor_config.json", "video_preprocessor_config.json", "tokenizer.json"):
            if not (model_path / name).is_file():
                raise FileNotFoundError(f"Cosmos checkpoint processor asset missing: {model_path / name}")
        result["tokenizer_type"] = str(model_path)
        for key in ("pretrained_model_name", "repository", "revision", "subdir"):
            result.pop(key, None)
    else:
        raise ValueError(f"Unsupported Cosmos tokenizer factory: {target}")
    result["config_variant"] = "hf"
    return result


def context_parallel_degree(world_size: int, query_heads: int, kv_heads: int) -> int:
    """Choose a compatible CP divisor, independently from weight sharding."""
    if any(type(value) is not int or value < 1 for value in (world_size, query_heads, kv_heads)):
        raise ValueError("World size and attention head counts must be positive integers")
    if query_heads % kv_heads:
        raise ValueError("Query heads must be divisible by KV heads")
    return next(degree for degree in range(min(world_size, 32), 0, -1)
                if world_size % degree == 0 and query_heads % degree == 0
                and kv_heads % degree == 0)


class CosmosFrameworkRuntime:
    """Mixin: callers supply explicit project-local assets and runtime parameters."""

    def _backbone_config_path(self, backbone):
        path = (self.framework_root / backbone).resolve()
        if not path.is_relative_to(self.framework_root):
            raise ValueError("Cosmos backbone config leaves framework_root")
        # Some released exports contain the canonical filename rather than
        # a cosmos3:// URI. Resolve this known architecture, never fetch a
        # different backbone or modify checkpoint files.
        if not path.is_file() and backbone == "Qwen3-VL-8B-Instruct.json":
            path = self.framework_root / "cosmos_framework/model/generator/reasoner/qwen3_vl/configs" / backbone
        if not path.is_file():
            raise FileNotFoundError(f"Local Cosmos backbone config missing: {path}")
        return path

    def _runtime_config(self) -> Path:
        payload = json.loads((self.model_path / "config.json").read_text(encoding="utf-8"))
        try:
            tokenizer = payload["model"]["config"]["tokenizer"]
        except (KeyError, TypeError) as exc:
            raise ValueError("DROID FD config has no model.config.tokenizer section") from exc
        tokenizer["bucket_name"] = ""
        tokenizer["vae_path"] = str(self.vae_path)
        text_tokenizer = self.text_tokenizer_path
        if not (text_tokenizer / "vocab.json").is_file():
            raise FileNotFoundError(f"Local Cosmos Qwen text tokenizer missing: {text_tokenizer}")
        vlm = payload["model"]["config"]["vlm_config"]
        vlm["tokenizer"] = local_tokenizer_config(vlm["tokenizer"], self.model_path, text_tokenizer)
        base = payload["model"]["config"]["vlm_config"]["model_instance"]["config"]["base_config"]
        if base.get("json_file") == "Qwen3-VL-8B-Instruct.json":
            base["json_file"] = str(self._backbone_config_path(base["json_file"]))
        # The export predates optional modality flags in this framework revision.
        # Fill only absent fields with upstream legacy-preserving defaults.
        import attrs
        from cosmos_framework.configs.base.defaults.model_config import DiffusionExpertConfig, OmniMoTModelConfig
        model_config = payload["model"]["config"]
        # Older exports omit newly optional schema fields. Only supply
        # upstream defaults for absent fields; preserve all exported choices.
        for key, value in attrs.asdict(OmniMoTModelConfig()).items():
            model_config.setdefault(key, value)
        expert = payload["model"]["config"]["diffusion_expert_config"]
        for key, value in attrs.asdict(DiffusionExpertConfig()).items():
            expert.setdefault(key, value)
        path = self.runtime_dir / "config.local.json"
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if dist.is_initialized():
            dist.barrier()
        return path

    def _load_pipeline(self) -> None:
        # Upstream enables training extras by default. This adapter never trains
        # and must not import private/data-collection integrations implicitly.
        os.environ["COSMOS_TRAINING"] = "0"
        os.environ["COSMOS_INTERNAL"] = "0"
        framework = str(self.framework_root)
        if framework not in sys.path:
            sys.path.insert(0, framework)
        try:
            from cosmos_framework.inference.common.init import init_output_dir, init_script
            init_script()
            from cosmos_framework.inference.args import OmniSampleOverrides, OmniSetupOverrides
            from cosmos_framework.inference.vision import decode_video_thwc_uint8
        except ImportError as exc:
            raise ImportError(
                "Cosmos Framework dependencies are unavailable; use the adapter's registered environment."
            ) from exc
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        checkpoint = json.loads((self.model_path / "config.json").read_text())
        from cosmos_framework.inference.common.public_model_config import load_model_config_from_hf_config
        restored = load_model_config_from_hf_config(checkpoint)
        backbone = restored["config"]["vlm_config"]["model_instance"]["config"]["base_config"]["json_file"]
        backbone_path = self._backbone_config_path(backbone)
        heads = json.loads(backbone_path.read_text())["text_config"]
        cp_degree = context_parallel_degree(world_size, heads["num_attention_heads"], heads["num_key_value_heads"])
        self.parallelism_contract = {"weight_shards": world_size, "context_parallel_degree": cp_degree,
                                     "query_heads": heads["num_attention_heads"], "kv_heads": heads["num_key_value_heads"]}
        setup = OmniSetupOverrides(
            checkpoint_path=str(self.model_path),
            config_file=str(self._runtime_config()),
            output_dir=self.runtime_dir / "outputs",
            guardrails=False,
            use_torch_compile=self.use_torch_compile,
            diffusion_cache=self.diffusion_cache,
            parallelism_preset="latency",
            dp_shard_size=world_size,
            cp_size=cp_degree,
            cfgp_size=1,
            max_num_seqs=getattr(self, "inference_batch_size", 1),
        )
        self._setup_args = setup.build_setup()
        init_output_dir(self._setup_args.output_dir)
        self._pipe = self._setup_args.get_inference_cls().create(self._setup_args)
        self._sample_overrides_cls = OmniSampleOverrides
        self._decode_video = decode_video_thwc_uint8
