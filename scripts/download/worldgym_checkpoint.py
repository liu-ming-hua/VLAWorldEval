"""Get the fixed WorldGym release; verify cached assets without importing Torch."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vlaworldeval.paths import project_root  # noqa: E402
from vlaworldeval.reference_asset import ensure_reference_asset  # noqa: E402
from vlaworldeval.gdown_client import download_google_drive  # noqa: E402

FILE_NAME = "mixed_openx_9robots_20frames_0p1actiondropout_580ksteps.pt"
GOOGLE_DRIVE_ID = "1uiRP2BuavapMsyP9Cbr25mi_ymk9SEJb"
# Measured from the release already used in the real WorldGym acceptance tests,
# not claimed to be a checksum published by the authors.
REFERENCE_SIZE = 9727371838
REFERENCE_SHA256 = "7acbf6702265e53909283ecdd4b3ddc2517f9fea460cf700369a076f2dfad182"


def main():
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path,
                        default=root / "checkpoints/world_models/worldgym/checkpoint" / FILE_NAME)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)

    def fetch(path):
        # Do not read/write cookies in the user's home cache. The released
        # checkpoint is public; TLS verification remains enabled.
        try:
            download_google_drive(GOOGLE_DRIVE_ID, path)
        except OSError as exc:
            raise ValueError("Google Drive request failed or timed out; partial data is retained for retry") from exc

    try:
        report = ensure_reference_asset(args.destination, root=root, size=REFERENCE_SIZE,
                                        sha256=REFERENCE_SHA256,
                                        source={"kind": "google_drive", "id": GOOGLE_DRIVE_ID},
                                        fetch=None if args.verify_only else fetch)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
