"""Run from a model/tool interpreter without installing the framework there."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vlaworldeval.assets import download_main  # noqa: E402

if __name__ == "__main__":
    download_main()
