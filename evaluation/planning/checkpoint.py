"""Complete-split preparation, never a private model inference loop."""
from __future__ import annotations

import json
from pathlib import Path

from vlaworldeval.assets import MANIFEST, verify_snapshot
from vlaworldeval.atomic_json import publish_json_exclusive
from evaluation.execution.transport import read_packet, write_packet
from evaluation.reporting.task_scores import _digest

IRASIM_TEST_COUNTS = (4562, 5243)


def prepare_worldgym_released_test(root, dataset, config_path, output, *, seeds=(0, 1, 2)):
    """Entire released sample test only; never label it the full OXE paper test."""
    import numpy as np
    from evaluation.datasets.native_reference import prepare_case
    from evaluation.planning.registry import project_path
    from vlaworldeval.asset_lock import asset_destination_lock
    root = Path(root).resolve()
    dataset, config_path, output = [project_path(root, str(p)) for p in (dataset, config_path, output)]
    if output.exists() or not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Use a fresh output and distinct nonnegative integer seeds")
    expected = {"000000002.mp4": "64d01ecc97a83e379474471d10787de6e56be0ed9c792dd3eea2f1a4920ae8d1",
                "000000002.npz": "71b51e11ad5d7a43d8258294bff0691d4522223b4684cf80ffea8b1253aa30a5"}
    actual = {p.name for p in dataset.iterdir() if p.suffix in {".mp4", ".npz"}}
    if actual != set(expected) or any(_digest(dataset/name) != sha for name, sha in expected.items()):
        raise ValueError("Pinned released sample-test membership/checksums differ; no substitute")
    with np.load(dataset/"000000002.npz", allow_pickle=False) as data:
        length = len(data["arr_0"])
    config = json.loads(config_path.read_text())
    if config.get("adapter") != "WorldGymAdapter" or config["action"]["horizon"] != 1 or config["video"]["frames_per_chunk"] != 1:
        raise ValueError("Whole22-frame sample requires native single-step replay, no tail padding")
    with asset_destination_lock(root, output):
        output.mkdir(parents=True)
        (output/"source_configs").mkdir()
        cases = []
        for seed in seeds:
            source_config = output/"source_configs"/f"seed{seed}.json"
            publish_json_exclusive(source_config, {**config, "seed": seed})
            identifier = f"worldgym_released_test2_all_actions_seed{seed}"
            target = output/"cases"/identifier
            prepare_case("worldgym", dataset, source_config, target, chunks=length-1, episode=2)
            cases.append({"id": identifier, "prepared": str(target.relative_to(root))})
        publish_json_exclusive(output/"plan.json", {"schema_version": 1,
            "scope": "Entire released WorldGym Bridge sample test:1 episode, all21 real future actions per seed; not full OXE paper validation",
            "cases": cases, "unavailable": {"full_oxe_paper_test": "No authoritative full paper test membership verified; nine training examples are not held-out test"}})
        publish_json_exclusive(output/"coverage.json", {"schema_version": 1, "source_commit": "709665b29f5c95124e7dbc05b0d4d47cd353de97",
            "source_sha256": expected, "expected_released_test_episodes": 1, "prepared_episodes": 1,
            "source_frames": length, "real_future_actions_per_seed": length-1, "seeds": list(seeds),
            "expected_cases": len(cases), "model_inference_performed": False,
            "full_paper_test_completed": False, "full_test_set_completed": False})
    return {"plan": str(output/"plan.json"), "released_test_episodes": 1, "cases": len(cases), "model_inference_performed": False}


def extract_irasim_test_annotations(transport, output):
    """Extract a closed test-annotation group from one verified gzip prefix.

    This does not claim the rest of the multipart archive or RGB is present.
    Only ordinary bounded JSON members with native paths are admitted.
    """
    import re
    import tarfile
    import numpy as np
    from vlaworldeval.paths import project_root
    from vlaworldeval.asset_lock import asset_destination_lock
    from vlaworldeval.assets import file_record
    transport, output = Path(transport).resolve(), Path(output).resolve()
    verified = verify_snapshot(transport)
    name = "languagetable_train_data.tar.gz.sfpart_aa"
    manifest = json.loads((transport/MANIFEST).read_text())
    if (verified["repository"] != "fangqi/IRASim" or verified["revision"] != "dfcbf85c27b2c5d041dbf5df3df272111e758b48"
            or set(manifest["files"]) != {name}):
        raise ValueError("Expected the pinned, checksum-verified first training-data shard")
    if output.exists() or not output.is_relative_to(project_root()):
        raise ValueError("Choose a new project-local annotation directory")
    expected_episodes, expected_clips = IRASIM_TEST_COUNTS
    prefix = "opensource_robotdata/languagetable/annotation/test/"
    records, clips, total_bytes, boundary = {}, [], 0, None
    with asset_destination_lock(project_root(), output):
        output.mkdir(parents=True)
        with tarfile.open(transport/name, mode="r|gz") as archive:
            for member in archive:
                if not member.name.startswith(prefix):
                    if records:
                        boundary = member.name
                        break
                    continue
                if member.isdir():
                    continue
                if (not member.isfile() or not re.fullmatch(re.escape(prefix)+r"[0-9]{6}\.json", member.name)
                        or not 0 < member.size <= 1024**2):
                    raise ValueError("Only ordinary bounded native JSON test members are permitted")
                stem = Path(member.name).stem
                if stem in records or len(records) >= expected_episodes:
                    raise ValueError("Duplicate/excessive test annotations")
                total_bytes += member.size
                if total_bytes > 100*1024**2:
                    raise ValueError("Test annotations exceed bounded extraction size")
                payload = archive.extractfile(member).read(member.size+1)
                annotation = json.loads(payload)
                actions = np.asarray(annotation["actions"], dtype=np.float32)
                if (annotation.get("episode_id") != stem or annotation.get("video_path") != f"videos/test/{stem}/rgb.mp4"
                        or annotation.get("latent_video_path") != f"latent_videos/test/{stem}/0.pt"
                        or actions.ndim != 2 or actions.shape[1] != 2 or not np.isfinite(actions).all()):
                    raise ValueError("Native episode identity/paths/XY action annotation differs")
                path = output/"annotation/test"/f"{stem}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                records[stem] = {"path": str(path.relative_to(output)), **file_record(path), "actions": len(actions),
                    "video_path": annotation["video_path"], "latent_video_path": annotation["latent_video_path"]}
                clips.extend({"episode": stem, "start": start, "source_frame_indices": list(range(start, start+16))}
                             for start in range(0, len(actions), 16) if start+15 < len(actions))
        if boundary is None or len(records) != expected_episodes or len(clips) != expected_clips:
            raise ValueError("Published test cardinality/group boundary differs; incomplete extraction is not a full split")
        report = {"schema_version": 1, "repository": verified["repository"], "revision": verified["revision"],
            "source_shard": manifest["files"][name], "source_transport": str(transport), "closed_group_next_member": boundary,
            "test_episodes": len(records), "official_short_clips": len(clips), "files": records, "clips": clips,
            "sampling": "upstream Dataset_2D:16frames, stride16 starts, len(actions) bounds;15 future actions",
            "full_archive_downloaded": False, "rgb_and_latents_acquired": False,
            "model_inference_performed": False, "full_test_set_completed": False}
        publish_json_exclusive(output/"coverage.json", report)
    return {key: report[key] for key in ("test_episodes", "official_short_clips", "full_archive_downloaded", "model_inference_performed")}


def audit_cosmos_droid_holdout(dataset, output):
    """Reconstruct20k-EMA loader's episode holdout and inspect local payloads.

    This is an audit, not a replay or evidence of independent scene holdout.
    Torch randperm is intentional: NumPy's permutation would select other IDs.
    Applies only to the community20k-EMA export whose model card documents it.
    """
    import torch
    import pyarrow.parquet as pq
    from evaluation.datasets.droid import CAMERAS
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    from vlaworldeval.paths import project_root
    if not output.is_relative_to(project_root()) or output.exists():
        raise ValueError("Choose a new project-local audit report")
    selected, metadata_hashes, missing = [], {}, set()
    for subset, count in (("success", 57639), ("failure", 14268)):
        directory = dataset/subset
        info_path = directory/"meta/info.json"
        info = json.loads(info_path.read_text())
        if info.get("total_episodes") != count or info.get("fps") != 15:
            raise ValueError("Dataset cardinality/FPS differs from20k-EMA training source")
        metadata_hashes[str(info_path)] = _digest(info_path)
        ids = torch.randperm(count, generator=torch.Generator().manual_seed(42)).tolist()[:int(round(count*.03))]
        wanted, rows = set(ids), {}
        columns = ["episode_index", "length", "data/chunk_index", "data/file_index"]
        columns += [f"videos/{key}/{field}" for key in CAMERAS.values() for field in ("chunk_index", "file_index")]
        for path in sorted((directory/"meta/episodes").glob("chunk-*/file-*.parquet")):
            metadata_hashes[str(path)] = _digest(path)
            for row in pq.read_table(path, columns=columns).to_pylist():
                index = row["episode_index"]
                if index in wanted:
                    if index in rows:
                        raise ValueError("Duplicate held-out episode metadata")
                    rows[index] = row
        for index in ids:
            if index not in rows:
                raise ValueError(f"Missing held-out metadata {subset}/{index}; do not shrink split")
            row = rows[index]
            paths = [directory/f"data/chunk-{row['data/chunk_index']:03d}/file-{row['data/file_index']:03d}.parquet"]
            for key in CAMERAS.values():
                prefix = f"videos/{key}"
                paths.append(directory/f"{prefix}/chunk-{row[prefix+'/chunk_index']:03d}/file-{row[prefix+'/file_index']:03d}.mp4")
            absent = [str(path.relative_to(dataset)) for path in paths if not path.is_file()]
            missing.update(absent)
            selected.append({"id": f"cosmos20k_{subset}_{index:06d}", "subset": subset,
                "episode": index, "length": row["length"],
                "complete16_action_chunks": max(0, (row["length"]-1)//16),
                "payload_paths": [str(p.relative_to(dataset)) for p in paths], "missing_files": absent})
    report = {"schema_version": 1, "model": "cosmos3_nano_droid_fd_20k_ema",
        "protocol": "per-root torch.randperm, seed42, round(N*.03), prefix validation",
        "source": "third_party/cosmos-framework/cosmos_framework/data/generator/action/datasets/cosmos3_action_lerobot.py:split_episode_ids",
        "training_overlap": "excluded episodes per model card; scene identity leakage remains possible; other checkpoints not covered",
        "expected_episodes": 2157, "episodes": selected, "metadata_sha256": metadata_hashes,
        "locally_available_episodes": sum(not c["missing_files"] for c in selected),
        "missing_files": sorted(missing), "payload_hash_verification": "pending; existence alone is not integrity evidence",
        "model_inference_performed": False, "full_test_set_completed": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    publish_json_exclusive(output, report)
    return {key: report[key] for key in ("expected_episodes", "locally_available_episodes", "model_inference_performed")}


def dreamdojo_eval100_selection(metadata_root):
    """Exact released restrict_len100 deterministic-uniform In-lab selection.

    Source action_conditioned.py passes restrict_len=num_samples. It is not a
    random/cherry-picked100 and not every sliding window in the raw corpus.
    """
    from evaluation.datasets.dreamdojo import INLAB_TASKS, episode_paths
    import numpy as np
    metadata_root = Path(metadata_root).resolve()
    verified = verify_snapshot(metadata_root)
    if (verified["repository"] != "nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1"
            or verified["revision"] != "9f68429ad12535d30d1aae33b7945c79c830c8c4"):
        raise ValueError("Official eval selection requires the pinned complete In-lab metadata")
    samples, audits = [], []
    for task in INLAB_TASKS:
        relative = Path("In-lab_Eval")/f"gr1_unified.{task}_robot"
        directory = metadata_root/relative
        episodes = [json.loads(line) for line in (directory/"meta/episodes.jsonl").read_text().splitlines() if line.strip()]
        if (not episodes or len({e["episode_index"] for e in episodes}) != len(episodes)
                or any(type(e["length"]) is not int or e["length"] < 1 for e in episodes)):
            raise ValueError("Invalid official ordered episode metadata")
        counts = np.asarray([max(0, e["length"]-98) for e in episodes], dtype=np.int64)
        cumulative = counts.cumsum()
        if cumulative[-1] < 10:
            raise ValueError("Official uniform sampling requires at least10 valid windows per task")
        audits.append({"task": task, "episodes": len(episodes), "valid49_frame_windows": int(cumulative[-1]),
                       "metadata_sha256": _digest(directory/"meta/episodes.jsonl")})
        for sample in range(10):
            local_index = int(sample/10 * int(cumulative[-1]))
            selected = int(np.searchsorted(cumulative, local_index, side="right"))
            start = int(local_index-(cumulative[selected-1] if selected else 0))
            episode = episodes[selected]["episode_index"]
            parquet, video = episode_paths(directory, episode)
            samples.append({"id": f"dreamdojo_inlab_{len(samples):03d}", "task": task,
                "dataset": str(relative), "episode": episode, "start": start, "chunks": 4,
                "parquet": str(parquet.relative_to(metadata_root)), "video": str(video.relative_to(metadata_root)),
                "official_dataset_local_index": local_index})
    return {"schema_version": 1, "source_snapshot": verified, "samples": samples, "tasks": audits,
        "expected_samples": 100, "protocol": {"num_frames": 49, "prefix_frames": 1, "source_stride": 2,
            "data_split": "full", "restrict_len": 100, "deterministic_uniform_sampling": True,
            "single_base_index": False, "seed_offsets_per_round": [0, 12, 24, 36]},
        "scope": "complete released100-sample In-lab evaluation protocol, not all raw sliding windows"}


def compare_dreamdojo_official_selection(metadata_root, output):
    """Run upstream step enumeration and100-sample indexing on real metadata.

    Payload lookup returns identities only; no image/learned model is mocked.
    This validates the selector separately from official encoding comparisons.
    """
    import sys
    import numpy as np
    from vlaworldeval.paths import project_root
    from adapters.world_models.dreamdojo import _source_cwd
    root = project_root()
    source = root/"third_party/dreamdojo"
    sys.path.insert(0, str(source))
    from groot_dreams.data.dataset import LeRobotSingleDataset
    from groot_dreams.dataloader import MultiVideoActionDataset
    from groot_dreams.groot_configs import construct_modality_config_and_transforms
    metadata_root, output = Path(metadata_root).resolve(), Path(output).resolve()
    if output.exists() or not output.is_relative_to(root):
        raise ValueError("Choose a new project-local selection comparison report")
    selected = dreamdojo_eval100_selection(metadata_root)
    config, _, _ = construct_modality_config_and_transforms(50, "gr1")

    class IndexView:
        def __init__(self, task, steps):
            self.task, self.steps = task, steps

        def __len__(self):
            return len(self.steps)

        def __getitem__(self, index):
            episode, start = self.steps[index]
            return {"task": self.task, "episode": int(episode), "start": int(start)}

    wrapper = MultiVideoActionDataset.__new__(MultiVideoActionDataset)
    wrapper.datasets = []
    wrapper.deterministic_uniform_sampling, wrapper.restrict_len = True, 100
    wrapper.t5_text_embeddings, wrapper.t5_text_mask = None, None
    for task in selected["tasks"]:
        dataset = LeRobotSingleDataset.__new__(LeRobotSingleDataset)
        dataset._dataset_path = metadata_root/"In-lab_Eval"/f"gr1_unified.{task['task']}_robot"
        dataset._trajectory_ids, dataset._trajectory_lengths = dataset._get_trajectories()
        dataset.modality_configs = config
        steps = np.asarray(dataset._get_all_steps(single_base_index=False), dtype=np.int64)
        wrapper.datasets.append(IndexView(task["task"], steps))
    with _source_cwd(source):
        for index, case in enumerate(selected["samples"]):
            if wrapper[index] != {k: case[k] for k in ("task", "episode", "start")}:
                raise ValueError(f"Upstream selection differs at official case{index}")
    report = {"schema_version": 1, "ok": True, "cases_compared": 100,
        "methods": ["LeRobotSingleDataset._get_all_steps", "MultiVideoActionDataset.__getitem__"],
        "identity_lookup_only": True, "model_inference_performed": False,
        "selection": selected, "source_sha256": {name: _digest(source/name) for name in (
            "groot_dreams/data/dataset.py", "groot_dreams/dataloader.py", "groot_dreams/groot_configs.py")}}
    output.parent.mkdir(parents=True, exist_ok=True)
    publish_json_exclusive(output, report)
    return {key: report[key] for key in ("ok", "cases_compared", "methods", "model_inference_performed")}


def prepare_dreamdojo_eval100(root, dataset, metadata, config, output, *, seeds=(0,), gpus=(0, 1, 2)):
    """Compile every officially selected case into the same native-study plan."""
    from evaluation.datasets.native_reference import prepare_case
    import torch
    torch.set_num_threads(4)
    root, dataset, output = Path(root).resolve(), Path(dataset).resolve(), Path(output).resolve()
    if output.exists() or not output.is_relative_to(root):
        raise ValueError("Choose a new project-local prepared study directory")
    if (not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds)
            or not gpus or len(set(gpus)) != len(gpus) or any(type(g) is not int or g not in range(4) for g in gpus)):
        raise ValueError("Declare distinct nonnegative seeds and allowed GPUs0..3")
    selected = dreamdojo_eval100_selection(metadata)
    verified = verify_snapshot(dataset)
    if any(verified[k] != selected["source_snapshot"][k] for k in ("repository", "revision")):
        raise ValueError("Payload and selection metadata must have the identical pinned identity")
    manifest = json.loads((dataset/MANIFEST).read_text())
    required = {case[key] for case in selected["samples"] for key in ("parquet", "video")}
    if not required <= set(manifest["files"]):
        raise ValueError("Incomplete official eval100 payload; no case substitution")
    output.mkdir(parents=True)
    cases, queues = [], {gpu: [] for gpu in gpus}
    for index, case in enumerate(selected["samples"]):
        first = None
        for seed in seeds:
            identifier = f"{case['id']}_seed{seed}"
            target = output/"cases"/identifier
            gpu = gpus[index % len(gpus)]
            if first is None:
                prepare_case("dreamdojo", dataset/case["dataset"], config, target,
                    start=case["start"], chunks=4, episode=case["episode"], gpu=gpu)
                first = target
            else:
                target.mkdir(parents=True)
                for file in ("input.npz", "reference.npz"):
                    (target/file).hardlink_to(first/file)
            runtime = json.loads((first/"config.json").read_text())
            runtime.update(seed=seed, gpus=[gpu])
            # prepare_case owns this mutable prepared config until plan freeze.
            from vlaworldeval.atomic_json import replace_owned_json
            if (target/"config.json").exists():
                replace_owned_json(target/"config.json", runtime)
            else:
                publish_json_exclusive(target/"config.json", runtime)
            entry = {"id": identifier, "prepared": str(target.relative_to(root))}
            cases.append(entry)
            queues[gpu].append(entry)
        print(f"Prepared official DreamDojo case{index+1}/100", flush=True)
    plan = {"schema_version": 1, "scope": selected["scope"], "cases": cases,
            "unavailable": {"other_domains": "This plan covers In-lab only; EgoDex/HV have separate protocols",
                "paper_numeric_identity": "Framework excludes anchors and removes duplicate chunk boundaries; not exact reproduction of author's aggregate implementation"}}
    publish_json_exclusive(output/"selection.json", selected)
    publish_json_exclusive(output/"plan.json", plan)
    publish_json_exclusive(output/"coverage.json", {"schema_version": 1, "expected_official_samples": 100,
        "prepared_samples": 100, "expected_cases": len(cases), "seeds": list(seeds),
        "payload": verified, "payload_manifest_sha256": _digest(dataset/MANIFEST),
        "selection_sha256": _digest(output/"selection.json"), "model_inference_performed": False,
        "full_test_set_completed": False})
    for gpu, entries in queues.items():
        publish_json_exclusive(output/f"plan_gpu{gpu}.json", {**plan, "cases": entries})
    return {"plan": str(output/"plan.json"), "official_samples": 100, "cases": len(cases), "model_inference_performed": False}


def prepare_iws_validation(root, dataset, config, output, *, seeds=(0, 1, 2)):
    """All ten published PushT validation episodes, short16 + autoregressive192.

    This is the complete public task validation split, NOT an assertion that the
    paper's undisclosed evaluation selection or all six task checkpoints are run.
    """
    import h5py
    from evaluation.datasets.native_reference import prepare_case
    root, dataset, output = Path(root).resolve(), Path(dataset).resolve(), Path(output).resolve()
    if output.exists() or not output.is_relative_to(root):
        raise ValueError("Choose a new project-local prepared study directory")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    verified = verify_snapshot(dataset)
    if (verified["repository"] != "yixuan1999/interactive-world-sim-data"
            or verified["revision"] != "6c3010b4d8c0fae8b683856ca1c550869d5bdb04"):
        raise ValueError("Full-split study cannot use the mini/debug corpus")
    expected = {f"pusht/val/episode_{i}.hdf5" for i in range(10)}
    manifest = json.loads((dataset/MANIFEST).read_text())
    actual = {name for name in manifest["files"] if name.startswith("pusht/val/")}
    if actual != expected:
        raise ValueError("Complete public validation requires every pinned PushT episode0..9")
    for name in sorted(expected):
        with h5py.File(dataset/name) as f:
            if len(f["obs/joint_pos"]) < 193:
                raise ValueError("Official 192-step evaluation needs 193 real frames; no padding")
    output.mkdir(parents=True)
    cases = []
    for episode in range(10):
        directory = output/"source"/f"episode{episode}"
        prepare_case("iws", dataset/f"pusht/val/episode_{episode}.hdf5", config, directory,
                     chunks=192, episode=episode)
        packet, reference = read_packet(directory/"input.npz"), read_packet(directory/"reference.npz")
        for mode, chunks in (("short", 16), ("autoregressive", 192)):
            for seed in seeds:
                identifier = f"iws_pusht_{episode}_{mode}_seed{seed}"
                target = output/"cases"/identifier
                target.mkdir(parents=True)
                sliced = {**packet, "actions": packet["actions"][:chunks]}
                write_packet(target/"input.npz", sliced)
                ref = {**reference, "input_sha256": _digest(target/"input.npz"),
                       "views": {k: v[:chunks+1] for k, v in reference["views"].items()},
                       "frame_indices": {k: v[:chunks+1] for k, v in reference["frame_indices"].items()}}
                write_packet(target/"reference.npz", ref)
                runtime = json.loads(directory.joinpath("config.json").read_text())
                runtime.update(seed=seed, gpus=[episode % 4])
                publish_json_exclusive(target/"config.json", runtime)
                cases.append({"id": identifier, "prepared": str(target.relative_to(root))})
    plan = {"schema_version": 1, "scope": "Complete public IWS PushT validation:10 episodes x2 horizons x declared seeds; future measured FK, not commanded-action evaluation",
            "cases": cases, "unavailable": {"paper_test_split": "No separately published paper test selection verified; official validation is not relabelled test",
                "other_iws_checkpoints": "This study covers only the installed pusht_cam1 checkpoint"}}
    publish_json_exclusive(output/"plan.json", plan)
    publish_json_exclusive(output/"coverage.json", {"schema_version": 1,
        "snapshot": verified, "snapshot_manifest_sha256": _digest(dataset/MANIFEST),
        "authoritative_episode_paths": sorted(expected), "expected_episodes": 10, "prepared_episodes": 10,
        "split": "official_public_validation", "checkpoint": "iws_pusht_cam1",
        "seeds": list(seeds), "horizons": {"short": 16, "autoregressive": 192},
        "expected_cases": len(cases), "model_inference_performed": False, "full_test_set_completed": False})
    for gpu in range(4):
        selected = [case for case in cases if int(case["id"].split("_")[2]) % 4 == gpu]
        publish_json_exclusive(output/f"plan_gpu{gpu}.json", {**plan, "cases": selected})
    return {"plan": str(output/"plan.json"), "prepared_episodes": 10, "cases": len(cases),
            "model_inference_performed": False}
