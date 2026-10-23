"""Explicit executable-pair inventory; never infer a Cartesian product."""
import json

from evaluation.manifests import load_pair_manifest
from evaluation.planning.catalog import load_model_catalog
from evaluation.planning.configuration import invalid
from evaluation.planning.registry import load_runtime_registry


def capabilities(root):
    catalog = load_model_catalog(root)
    registry = load_runtime_registry(root)
    pairs = []
    for path in sorted((root / "configs/pairs").glob("*.json")):
        pair = load_pair_manifest(path)
        profile = registry.pair_profile(pair)
        pairs.append({
            "pair_id": pair.pair_id, "manifest": str(path.relative_to(root)),
            "runtime_profile": profile["id"], "dataset_loader": profile["dataset"],
            "vla": pair.vla.adapter, "world_model": pair.world_model.adapter,
            "environments": {role: registry.models[role][adapter]["environment"] for role, adapter in
                             (("vlas", pair.vla.adapter), ("world_models", pair.world_model.adapter))},
            "native_action_shape": [pair.vla.action.horizon, pair.vla.action.dimension],
            "wm_action_shape": [pair.world_model.action.horizon, pair.world_model.action.dimension],
            "action_bridge": pair.action_bridge.name,
            "state_provider": pair.state_provider.type,
            "supported_evaluation_modes": ["wm_replay", *profile.get("evaluation_modes", ["policy_loop"])],
            "raw_wm_views": list(pair.world_model.output_views),
            "artifact_views": list(pair.video_output_views),
            "video": pair.to_dict()["world_model"]["video"],
            "status": "registered_not_executed_by_this_command",
            "declared_validation_status": pair.metadata.get("status", "unspecified"),
        })
    used = {"vlas": {p["vla"] for p in pairs}, "world_models": {p["world_model"] for p in pairs}}
    native_profiles = []
    for path in sorted((root / "configs/replay").glob("*.json")):
        config = json.loads(path.read_text())
        native_profiles.append({"profile": str(path.relative_to(root)), "world_model": config["adapter"],
                                "environment": config["environment"], "action": config["action"],
                                "video": config["video"], "output_views": config["output_views"],
                                "policy_pair_supported": False, "execution_performed": False})
    model_support = {}
    for group, role in (("vlas", "vla"), ("world_models", "world_model")):
        model_support[group] = {}
        for name, entry in catalog[group].items():
            declared = entry.get("integration", {"level": "unclassified", "scope": "Not classified",
                                                  "reason": "No integration declaration", "supported_pair_ids": []})
            registered = [p["pair_id"] for p in pairs if p[role] == entry["adapter"]]
            if set(declared["supported_pair_ids"]) - set(registered) or (
                    declared["level"] == "incompatible" and registered):
                invalid(root / "configs/model_catalog.json", f"{group}.{name}.integration",
                        "Declared support contradicts the registered adapter pairs")
            model_support[group][name] = {**declared, "registered_pair_ids": registered,
                "native_replay_profiles": [p["profile"] for p in native_profiles
                    if group == "world_models" and p["world_model"] == entry["adapter"]],
                "declared_validation_status": entry["status"], "execution_performed": False,
                "quality_evaluated": False}
    return {
        "schema_version": 1, "model_loading": False, "execution_performed": False,
        "quality_evaluated": False, "arbitrary_composition_supported": False,
        "catalogued_counts": {group: len(catalog[group]) for group in used},
        "executable_pairs": pairs,
        "native_replay_profiles": native_profiles,
        "model_support": model_support,
        "catalogued_without_unified_pair": {group: [name for name, entry in catalog[group].items()
                                                    if entry["adapter"] not in used[group]] for group in used},
        "note": "Registered interfaces are not executed evidence or policy-quality rankings. See actual acceptance/results reports.",
    }
