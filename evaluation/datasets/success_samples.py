"""Fixed DROID candidate windows, independent of generated task outcomes."""
from copy import deepcopy


def candidate_suite(template, *, episode, length, start=15, source_horizon=16):
    """Use all complete chunks; keep the discarded tail explicit, never pad."""
    if any(type(value) is not int for value in (episode, length, start, source_horizon)):
        raise ValueError("Window values must be integers")
    if episode < 0 or start < 15 or source_horizon not in (15, 16):
        raise ValueError("Require real N1.7 prehistory and a registered DROID horizon")
    chunks = (length - start - 1) // source_horizon
    if chunks < 1:
        raise ValueError("Candidate has no complete action chunk")
    suite = deepcopy(template)
    suite["suite_id"] = f"success_candidates_{template['suite_id']}_ep{episode:06d}"
    suite["dataset"].update(episodes=[episode], start_frame=start)
    suite["seeds"] = [0]
    suite["output_root"] = f"results/experiments/{suite['suite_id']}"
    provider = template["experiments"][-1]["state_provider"]
    suite["experiments"] = [
        {"id": "real_action_control", "mode": "wm_replay",
         "action_source": "dataset_pose_derived", "state_provider": "none", "chunks": chunks},
        {"id": "policy_candidate", "mode": "policy_loop", "action_source": "vla",
         "state_provider": provider, "chunks": chunks},
    ]
    suite["metadata"] = {"purpose": "fixed_success_candidate_screening",
        "task_success": None, "selection_uses_generated_outcomes": False,
        "source_horizon": source_horizon, "source_end_frame": start + chunks * source_horizon,
        "discarded_tail_frames": length - 1 - start - chunks * source_horizon,
        "train_overlap": "unknown", "dataset_success_split_is_not_policy_success": True}
    return suite
