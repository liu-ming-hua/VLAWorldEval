"""Project-local runtime caches, independent of the user's small home disk."""
from pathlib import Path


def runtime_cache_environment(root: Path) -> dict[str, str]:
    cache = root.resolve() / ".cache"
    return {key: str(cache / suffix) for key, suffix in {
        "XDG_CACHE_HOME": "",
        "HF_HOME": "huggingface",
        "HF_HUB_CACHE": "huggingface/hub",
        "HF_DATASETS_CACHE": "huggingface/datasets",
        "MODELSCOPE_CACHE": "modelscope",
        "TRANSFORMERS_CACHE": "huggingface/hub",
        "MPLCONFIGDIR": "matplotlib",
        "TFHUB_CACHE_DIR": "tfhub",
        "NUMBA_CACHE_DIR": "numba",
        "WANDB_CACHE_DIR": "wandb/cache",
        "WANDB_DATA_DIR": "wandb/data",
        "TORCH_HOME": "torch",
        "TORCH_EXTENSIONS_DIR": "torch_extensions",
        "TORCHINDUCTOR_CACHE_DIR": "torch_inductor",
        "TRITON_CACHE_DIR": "triton",
        "CUDA_CACHE_PATH": "cuda",
        "JAX_COMPILATION_CACHE_DIR": "jax",
        "OPENPI_DATA_HOME": "openpi",
    }.items()}
