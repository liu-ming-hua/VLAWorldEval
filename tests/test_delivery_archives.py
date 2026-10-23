"""Published model guides are allowed; runtime payloads remain excluded."""
import io
import tarfile
import zipfile

import pytest

from scripts.acceptance.public_delivery import check_archives


def archives(tmp_path, extra):
    wheel = tmp_path / "package.whl"
    sdist = tmp_path / "package.tar.gz"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in ("configs/runtime_registry.json", "configs/kinematics/droid_panda_link8_v1.json",
                     "configs/benchmarks/native_domains_v1.json", "environment/requirements/video_metrics.txt",
                     "environment/registry.toml", "LICENSE", "LICENSE_SCOPE.md"):
            archive.writestr("vlaworldeval/" + name, "{}")
    with tarfile.open(sdist, "w:gz") as archive:
        for name in ("scripts/setup/core.sh", "scripts/setup/common.sh", "environment/locks/core-py311.txt",
                     "environment/locks/build-tools-py311.txt", extra):
            member = tarfile.TarInfo("vlaworldeval-0.1.0/" + name)
            data = b"example"
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return wheel, sdist


def test_model_guide_is_not_runtime_model_code(tmp_path):
    assert check_archives(*archives(tmp_path, "docs/models/DREAMDOJO.md"))["sdist_files"] == 5


@pytest.mark.parametrize("path", ["models/runtime.py", "docs/models/weights.pt",
                                      "docs/models/runtime.py", "third_party/model/runtime.py",
                                      "results/rollout.mp4", "docs/assets/demos/other.mp4"])
def test_runtime_payload_still_rejected(tmp_path, path):
    with pytest.raises(ValueError, match="Unexpected runtime/model/data asset"):
        check_archives(*archives(tmp_path, path))


@pytest.mark.parametrize("filename", ["groot-cosmos.mp4", "cosmos-droid-task.mp4", "cosmos-droid-long.mp4"])
def test_only_named_presentation_video_is_allowed(tmp_path, filename):
    assert check_archives(*archives(tmp_path, "docs/assets/demos/"+filename))["sdist_files"] == 5
