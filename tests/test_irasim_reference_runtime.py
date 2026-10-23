"""Real bundled RGB/latent preparation checks in IRASim's optional environment."""
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("decord")
pytest.importorskip("torchvision")


def test_native_source_loader_and_initial_latent_only_packet_on_actual_bundled_media(tmp_path):
    from evaluation.datasets.irasim_reference import load_episode, official_video_frames, write_short_case, write_long_case
    from evaluation.execution.transport import read_packet
    root = Path(__file__).resolve().parents[1]
    dataset = root/"datasets/irasim/languagetable"
    repo = root/"third_party/irasim"
    if not dataset.is_dir() or not repo.is_dir():
        pytest.skip("Official optional media/checkout not installed")
    torch.set_num_threads(4)
    annotation = json.loads((dataset/"annotation/test/example.json").read_text())
    annotation.update(episode_id="000000", video_path="videos/test/000000/rgb.mp4",
                      latent_video_path="latent_videos/test/000000/0.pt")
    native = tmp_path/"native"
    path = native/"annotation/test/000000.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(annotation))
    for old, new in (("videos/test/example.mp4", annotation["video_path"]),
                     ("latent_videos/test/example.pt", annotation["latent_video_path"])):
        target = native/new
        target.parent.mkdir(parents=True)
        shutil.copyfile(dataset/old, target)
    data = load_episode(native, "000000")
    frames = official_video_frames(repo, native, data[0], range(16))
    assert frames.shape == (16, 288, 512, 3) and frames.dtype == np.uint8
    sample = {"episode": "000000", "start": 0, "source_frame_indices": list(range(16))}
    config = json.loads((root/"configs/replay/irasim_native_v1.json").read_text())
    target = tmp_path/"prepared"
    write_short_case(native, repo, config, target, sample, data, {"fixture_relocated_bundled_media": True})
    packet, reference = read_packet(target/"input.npz"), read_packet(target/"reference.npz")
    assert packet["actions"].shape == (1, 15, 2)
    np.testing.assert_array_equal(packet["actions"][0], np.asarray(annotation["actions"][:15], np.float32))
    np.testing.assert_array_equal(packet["reset_args"]["value"]["latent"], data[2][0].numpy())
    assert packet["reset_args"]["value"]["latent"].shape == (4, 36, 64)
    assert packet["source"]["future_latents_sent_to_worker"] is False
    np.testing.assert_array_equal(reference["views"]["primary"], frames)
    np.testing.assert_array_equal(packet["initial_views"]["primary"], frames[0])
    long_config = {**config, "video": {**config["video"], "final_chunk_valid_frames": (len(data[1])-1)%15+1}}
    long_sample = {"episode": "000000", "start": 0, "source_frame_indices": list(range(len(data[1])+1))}
    write_long_case(native, repo, long_config, tmp_path/"long", long_sample, data, {"fixture_relocated_bundled_media": True})
    long_packet = read_packet(tmp_path/"long/input.npz")
    long_ref = read_packet(tmp_path/"long/reference.npz")
    assert long_packet["actions"].shape == ((len(data[1])+14)//15, 15, 2)
    assert len(long_ref["views"]["primary"]) == len(data[1])+1
    np.testing.assert_array_equal(long_packet["actions"].reshape(-1, 2)[:len(data[1])], data[1])
    assert long_packet["source"]["padded_predictions_scored"] is False
    from evaluation.manifests import VideoSpec
    assert VideoSpec.from_dict(long_config["video"]).frame_count(len(long_packet["actions"])) == len(data[1])+1
    assert not torch.cuda.is_initialized()
