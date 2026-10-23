"""Reference compression changes storage, not pixels, actions or dtype."""
import zipfile

import numpy as np
import pytest

from evaluation.execution.transport import read_packet, write_packet


@pytest.mark.parametrize("compressed", [False, True])
def test_packet_compression_is_lossless_and_explicit(tmp_path, compressed):
    path = tmp_path/"packet.npz"
    payload = {"views": {"camera": np.zeros((17, 24, 32, 3), np.uint8)},
               "actions": np.arange(160, dtype=np.float32).reshape(16, 10), "metadata": [None, True, "native"]}
    write_packet(path, payload, compressed=compressed)
    value = read_packet(path)
    np.testing.assert_array_equal(value["views"]["camera"], payload["views"]["camera"])
    np.testing.assert_array_equal(value["actions"], payload["actions"])
    assert value["views"]["camera"].dtype == np.uint8 and value["metadata"] == payload["metadata"]
    with zipfile.ZipFile(path) as archive:
        assert {z.compress_type for z in archive.infolist()} == {zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED}


def test_unfrozen_reference_can_be_atomically_reencoded_without_semantic_change(tmp_path):
    path = tmp_path/"reference.npz"
    write_packet(path, {"frames": np.ones((17, 24, 32, 3), np.uint8), "input_sha256": "fixture"})
    raw_size = path.stat().st_size
    write_packet(path, read_packet(path), compressed=True)
    assert path.stat().st_size < raw_size
    assert read_packet(path)["input_sha256"] == "fixture"


def test_invalid_compression_option_does_not_write(tmp_path):
    with pytest.raises(ValueError):
        write_packet(tmp_path/"not-written.npz", {}, compressed="lossy")
    assert not (tmp_path/"not-written.npz").exists()
