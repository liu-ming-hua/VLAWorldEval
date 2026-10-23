---
license: openmdw-1.1
---

# DROID: Distributed Robot Interaction Dataset

## Dataset Summary

DROID (Distributed Robot Interaction Dataset) is a large-scale "in-the-wild" robot manipulation dataset containing 76K teleoperated demonstration trajectories — approximately 350 hours of interaction data — collected across 564 unique scenes, 86 tasks, and 52 buildings over the course of 12 months. The data was collected by 50 data collectors at 18 labs across 13 institutions in North America, Asia, and Europe, on a single shared, open-source robot hardware platform. Every DROID episode contains three synchronized stereo RGB camera streams, camera calibration data, depth information, low-level robot state and control commands, and up to three natural-language task instructions.

## Dataset Details

### Dataset Description

DROID was designed to enable training of generalizable robot manipulation policies by substantially increasing the diversity of scenes, tasks, objects, viewpoints, and interaction locations over prior real-world robot manipulation datasets. Data was collected on a single uniform robot hardware stack (a Franka Panda 7-DoF arm with a Robotiq 2F-85 gripper) so that demonstrations gathered at any of the 18 collection labs are comparable and reproducible.

Each data collection session begins with moving the portable robot platform to a new scene. The data collector chooses third-person camera views that capture interesting behaviors in the scene, performs extrinsic camera calibration using a checkerboard and the OpenCV calibration algorithm, then enters all potential tasks for the scene into the data-collection GUI either by selecting from a list of options or by typing free-form task instructions. During data collection the GUI prompts the data collector with a randomly sampled task from this list for each new episode; this ensures that data collection is not biased toward easier tasks or closer objects. The GUI also periodically prompts the data collector to perform randomly sampled "scene augmentations" such as moving the robot base, re-calibrating the 3rd-person cameras, changing room lighting, or adding/removing items in the scene. A data collector typically gathers up to 100 trajectories (about 20 minutes of interaction data) per scene before moving on to a new scene.
The paper reports that co-training policies with DROID boosts policy performance, robustness, and generalization over state-of-the-art approaches that leverage existing large-scale robot manipulation datasets by 20% on average across the 6 evaluation tasks and 4 evaluation locations described in the paper.

This version converts **raw** DROID data to the LeRobotDataset v3.0 standard. LeRobotDataset v3.0 uses file-based Parquet and MP4 shards plus structured metadata, replacing per-episode files with chunked files that can contain multiple episodes.

This dataset is ready for commercial or non-commercial uses.

## Dataset Owner(s)

NVIDIA Corporation

## Dataset Creation Date

2026-04-12

## Version:
v1.0 <br>

## License/Terms of Use

This dataset is released under the [OpenMDW1.1](https://openmdw.ai/)

### Dataset Sources

- Repository: https://huggingface.co/datasets/nvidia/Cosmos3-DROID
- DROID project page: https://droid-dataset.github.io/
- DROID paper: https://arxiv.org/abs/2403.12945
- DROID dataset doc: https://droid-dataset.github.io/droid/the-droid-dataset
- LeRobotDataset v3.0 documentation: https://huggingface.co/docs/lerobot/lerobot-dataset-v3

## Intended Use

This dataset is intended for research and development in:

- robot policy learning
- robot forward dynamics model learning
- robot inverse dynamics model learning

## Dataset Characterization

**Data Collection Method**

[Manually-Collected]

**Labeling Method**

[Manually-Collected]

## Dataset Format

The episodes are organized into two top-level directories: `success/` and `failure/`, each representing a separate LeRobotDataset v3.0 dataset. For example, the directory structure of `success/` is shown below:

```text
success/
├── data/
│   └── chunk-000/
│       └── file-*.parquet
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.parquet
│   └── episodes/
│       └── chunk-000/
│           └── file-*.parquet
└── videos/
    ├── observation.image.exterior_image_1_left/
    │   └── chunk-000/
    │       └── file-*.mp4
    ├── observation.image.exterior_image_2_left/
    │   └── chunk-000/
    │       └── file-*.mp4
    └── observation.image.wrist_image_left/
        └── chunk-000/
            └── file-*.mp4
```

### Dataset Quantification

The following values are taken from each LeRobotDataset's `meta/info.json`.

| Type | Episodes | Frames | Task IDs | FPS |
|---|---:|---:|---:|---:|
| `success` | 57,639 | 18,691,281 | 53,086 | 15 |
| `failure` | 14,268 | 3,721,431 | 52 | 15 |
| **Total** | **71,907** | **22,412,712** | **53,138** | **15** |

This dataset is converted directly from the **raw** DROID data. The statistics above differs from the DROID's **RLDS** data and the statistics reported in the paper (i.e. 76K success episodes and 16K failure episodes).

Total Data Storage: 707 GB

### Features

The following feature schema is declared in `meta/info.json`.

| Feature | Type | Shape / Details |
|---|---|---|
| `observation.image.exterior_image_1_left` | `video` | RGB, 640 x 360, AV1 MP4, 15 FPS, no audio |
| `observation.image.exterior_image_2_left` | `video` | RGB, 640 x 360, AV1 MP4, 15 FPS, no audio |
| `observation.image.wrist_image_left` | `video` | RGB, 640 x 360, AV1 MP4, 15 FPS, no audio |
| `observation.state.joint_positions` | `float32` | 7-dimensional joint position state |
| `observation.state.joint_velocities` | `float32` | 7-dimensional joint velocity state |
| `observation.state.joint_torques_computed` | `float32` | 7-dimensional computed torque state |
| `observation.state.motor_torques_measured` | `float32` | 7-dimensional measured torque state |
| `observation.state.cartesian_position` | `float32` | 6-dimensional cartesian position state |
| `observation.state.gripper_position` | `float32` | 1-dimensional gripper position state |
| `action.joint_position` | `float32` | 7-dimensional joint position action |
| `action.joint_velocity` | `float32` | 7-dimensional joint velocity action |
| `action.cartesian_position` | `float32` | 7-dimensional cartesian position action |
| `action.cartesian_velocity` | `float32` | 7-dimensional cartesian velocity action |
| `action.gripper_position` | `float32` | 1-dimensional gripper position action |
| `action.gripper_velocity` | `float32` | 7-dimensional gripper velocity action |
| `timestamp` | `float32` | Frame timestamp |
| `frame_index` | `int64` | Frame index within episode |
| `episode_index` | `int64` | Episode index |
| `index` | `int64` | Global frame index |
| `task_index` | `int64` | Task index |

### File Format

- Frame-level state/action data: Apache Parquet under `data/`.
- Episode metadata: chunked Parquet under `meta/episodes/`.
- Dataset schema and statistics: `meta/info.json` and `meta/stats.json`.
- Task metadata: `meta/tasks.parquet`.
- Video observations: AV1-encoded MP4 files under `videos/`.

## Loading

Authenticate with Hugging Face if required, then download or stream the dataset using LeRobot-compatible tooling.

```python
from huggingface_hub import snapshot_download

repo_dir = snapshot_download(
    repo_id="nvidia/Cosmos3-DROID",
    repo_type="dataset",
)

print(repo_dir)
```

Example LeRobot usage:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(f"{repo_dir}/success")
print(dataset.num_episodes)
print(dataset.meta.info["features"].keys())
```

## Dataset Creation

### Source Data

The source data is derived from DROID, a large-scale "in-the-wild" robot manipulation dataset for generalizable robot manipulation policy learning. The DROID paper describes high-quality human-teleoperated demonstration data collected on a shared Franka Panda hardware platform across 18 labs and 13 institutions over 12 months.

### Conversion

This repository converts **raw** DROID data into LeRobotDataset v3.0. In LeRobotDataset v3.0, data and video frames from multiple episodes are grouped into larger shard files, while metadata records episode boundaries, task IDs, feature schemas, statistics, and path templates.

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. Developers should work with their internal developer teams to ensure this dataset meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

## Citation

If you use this dataset, please cite the original DROID paper:

```bibtex
@article{khazatsky2024droid,
 author  = {Alexander Khazatsky and Karl Pertsch and Suraj Nair and Ashwin Balakrishna and Sudeep Dasari and Siddharth Karamcheti and Soroush Nasiriany and Mohan Kumar Srirama and Lawrence Yunliang Chen and Kirsty Ellis and Peter David Fagan and Joey Hejna and Masha Itkina and Marion Lepert and Yecheng Jason Ma and Patrick Tree Miller and Jimmy Wu and Suneel Belkhale and Shivin Dass and Huy Ha and Arhan Jain and Abraham Lee and Youngwoon Lee and Marius Memmel and Sungjae Park and Ilija Radosavovic and Kaiyuan Wang and Albert Zhan and Kevin Black and Cheng Chi and Kyle Beltran Hatch and Shan Lin and Jingpei Lu and Jean Mercat and Abdul Rehman and Pannag R Sanketi and Archit Sharma and Cody Simpson and Quan Vuong and Homer Rich Walke and Blake Wulfe and Ted Xiao and Jonathan Heewon Yang and Arefeh Yavary and Tony Z. Zhao and Christopher Agia and Rohan Baijal and Mateo Guaman Castro and Daphne Chen and Qiuyu Chen and Trinity Chung and Jaimyn Drake and Ethan Paul Foster and Jensen Gao and Vitor Guizilini and David Antonio Herrera and Minho Heo and Kyle Hsu and Jiaheng Hu and Muhammad Zubair Irshad and Donovon Jackson and Charlotte Le and Yunshuang Li and Kevin Lin and Roy Lin and Zehan Ma and Abhiram Maddukuri and Suvir Mirchandani and Daniel Morton and Tony Nguyen and Abigail O'Neill and Rosario Scalise and Derick Seale and Victor Son and Stephen Tian and Emi Tran and Andrew E. Wang and Yilin Wu and Annie Xie and Jingyun Yang and Patrick Yin and Yunchu Zhang and Osbert Bastani and Glen Berseth and Jeannette Bohg and Ken Goldberg and Abhinav Gupta and Abhishek Gupta and Dinesh Jayaraman and Joseph J Lim and Jitendra Malik and Roberto Martín-Martín and Subramanian Ramamoorthy and Dorsa Sadigh and Shuran Song and Jiajun Wu and Michael C. Yip and Yuke Zhu and Thomas Kollar and Sergey Levine and Chelsea Finn},
 title   = {{DROID}: A Large-Scale In-The-Wild Robot Manipulation Dataset},
 journal = {arXiv preprint arXiv:2403.12945},
 year    = {2024},
}
```

## References

- DROID project page: https://droid-dataset.github.io/
- DROID paper: https://arxiv.org/abs/2403.12945
- DROID dataset doc: https://droid-dataset.github.io/droid/the-droid-dataset
- DROID hardware GitHub repository: https://github.com/droid-dataset/droid
- DROID policy learning GitHub repository: https://github.com/droid-dataset/droid_policy_learning
- DROID updated annotations and improved calibrations on Hugging Face: https://huggingface.co/KarlP/droid
- LeRobotDataset v3.0 documentation: https://huggingface.co/docs/lerobot/lerobot-dataset-v3
