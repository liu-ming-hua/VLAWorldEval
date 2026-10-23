"""A shared EEF observation loader must not mislabel joint-command replay."""
from scripts.checks.cosmos_joint_droid import joint_replay_source


def test_joint_provenance_replaces_legacy_labels_without_mutating_source():
    legacy = {"source_frames_per_chunk": 16,
              "replay_action_source": "measured EEF pose transitions + commanded gripper",
              "episode_index": 351, "start_frame": 15, "end_frame": 239}
    actual = joint_replay_source(legacy)
    assert actual["source_frames_per_chunk"] == 32
    assert actual["replay_action_source"] == "recorded action.joint_position + action.gripper_position"
    assert actual["recurrent_state_source"] == "last_command_target_proxy_not_predicted_state"
    assert actual["end_frame"] == 239 and actual["episode_index"] == 351
    assert legacy["source_frames_per_chunk"] == 16
    assert legacy["replay_action_source"].startswith("measured EEF")
