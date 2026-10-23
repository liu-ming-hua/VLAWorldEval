"""Cooperating native studies never need duplicate GPU-resident models."""
import pytest

from evaluation.execution.admission import native_gpu_lease, ResourceAdmissionError


def test_same_gpu_is_exclusive_and_other_gpu_is_independent(tmp_path):
    with native_gpu_lease(tmp_path, [0]):
        with pytest.raises(ResourceAdmissionError, match="GPU 0"):
            with native_gpu_lease(tmp_path, [0]):
                pytest.fail("Second resident model admitted")
        with native_gpu_lease(tmp_path, [1]):
            pass
    with native_gpu_lease(tmp_path, [0]):
        pass


def test_partial_multi_gpu_acquisition_is_released(tmp_path):
    with native_gpu_lease(tmp_path, [1]):
        with pytest.raises(ResourceAdmissionError):
            with native_gpu_lease(tmp_path, [0, 1]):
                pass
        with native_gpu_lease(tmp_path, [0]):
            pass


def test_explicit_higher_gpu_id_is_supported_without_changing_defaults(tmp_path):
    with native_gpu_lease(tmp_path, [7]):
        with pytest.raises(ResourceAdmissionError, match="GPU 7"):
            with native_gpu_lease(tmp_path, [7]):
                pass


def test_exception_releases_lease_without_unlinking_lock(tmp_path):
    with pytest.raises(RuntimeError):
        with native_gpu_lease(tmp_path, [0]):
            raise RuntimeError("worker failed")
    before = {p.name: p.stat().st_ino for p in (tmp_path/".cache/vlaworldeval/asset_locks").iterdir()}
    with native_gpu_lease(tmp_path, [0]):
        pass
    assert before == {p.name: p.stat().st_ino for p in (tmp_path/".cache/vlaworldeval/asset_locks").iterdir()}


@pytest.mark.parametrize("gpus", [[True], [1.5], [-1], [0, 0], (0,), ["0"]])
def test_invalid_gpu_lease_fails_before_loading(gpus, tmp_path):
    with pytest.raises(ValueError):
        with native_gpu_lease(tmp_path, gpus):
            pytest.fail("Invalid devices admitted")
