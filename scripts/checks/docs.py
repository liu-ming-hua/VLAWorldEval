"""Check current repository documentation without network access."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from evaluation.validation.documentation import validate_documentation
    print(validate_documentation(ROOT))
