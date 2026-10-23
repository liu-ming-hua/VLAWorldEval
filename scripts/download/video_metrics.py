"""Verified original TorchVision VGG16 transport through the HF mirror."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


if __name__ == "__main__":
    from vlaworldeval.assets import download_snapshot
    print(download_snapshot("timm/vgg16.tv_in1k", ROOT / "checkpoints/evaluators/lpips_vgg",
        root=ROOT, revision="b8d8aa2dd860af9233c8c67385a8097fd6c35d3f",
        patterns=["model.safetensors", "config.json"]))
