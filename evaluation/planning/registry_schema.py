"""Weight-free runtime declaration validation with field-level diagnostics."""
import json
import math

from evaluation.planning.configuration import invalid


def load_runtime_data(path, validate_bindings):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        invalid(path, "$", str(exc))

    def fields(value, location, required, optional=None):
        if not isinstance(value, dict):
            invalid(path, location, "Expected an object")
        for key in sorted(value.keys() - required.keys() - (optional or {}).keys()):
            invalid(path, f"{location}.{key}", "Unknown field")
        for key, types in {**required, **(optional or {})}.items():
            field = f"{location}.{key}"
            if key not in value:
                if key in required:
                    invalid(path, field, "Required field is missing")
            elif type(value[key]) not in types:
                invalid(path, field, "Unexpected field type")

    def text(value, location):
        if not isinstance(value, str) or not value.strip():
            invalid(path, location, "Expected a nonempty string")

    def strings(value, location, *, nonempty=False):
        if not isinstance(value, list) or (nonempty and not value):
            invalid(path, location, "Expected a nonempty list" if nonempty else "Expected a list")
        for i, item in enumerate(value):
            text(item, f"{location}[{i}]")
        if len(set(value)) != len(value):
            invalid(path, location, "Duplicate list entries")

    def module(value, location, prefix):
        text(value, location)
        if not value.startswith(prefix + ".") or not all(part.isidentifier() for part in value.split(".")):
            invalid(path, location, f"Expected an import path below {prefix}")

    def bindings(value, location, *, paths_only=False):
        try:
            validate_bindings(value, paths_only=paths_only)
        except ValueError as exc:
            invalid(path, location, str(exc))

    fields(data, "$", {"schema_version": (int,), "models": (dict,), "datasets": (dict,), "pairs": (list,)})
    if data["schema_version"] != 1:
        invalid(path, "$.schema_version", "Expected schema version 1")
    models, datasets = data["models"], data["datasets"]
    fields(models, "$.models", {"vlas": (dict,), "world_models": (dict,)})
    for group, definitions in models.items():
        for adapter, definition in definitions.items():
            location = f"$.models.{group}.{adapter}"
            if not adapter.isidentifier():
                invalid(path, location, "Invalid adapter class name")
            fields(definition, location, {"module": (str,), "environment": (str,),
                "distributed": (bool,), "kwargs": (dict,)}, {"assets": (dict,)})
            module(definition["module"], location + ".module", "adapters")
            text(definition["environment"], location + ".environment")
            bindings(definition["kwargs"], location + ".kwargs")
            bindings(definition.get("assets", {}), location + ".assets", paths_only=True)
    allowed_context = {"root", "episode", "start", "chunks", "horizon", "fps", "instruction", "replay"}
    for identifier, definition in datasets.items():
        location = f"$.datasets.{identifier}"
        text(identifier, location)
        fields(definition, location, {"module": (str,), "loader": (str,), "arguments": (dict,)})
        module(definition["module"], location + ".module", "evaluation")
        if not definition["loader"].isidentifier():
            invalid(path, location + ".loader", "Invalid loader name")
        for argument, source in definition["arguments"].items():
            if not argument.isidentifier() or not isinstance(source, str) or source not in allowed_context:
                invalid(path, location + ".arguments." + argument, "Invalid dataset context binding")
    identities, profile_ids = set(), set()
    sampling = {"image_size", "resolution", "torch_compile", "diffusion_cache", "num_inference_steps",
                "guidance", "shift"}
    for i, profile in enumerate(data["pairs"]):
        location = f"$.pairs[{i}]"
        fields(profile, location, {"id": (str,), "vlas": (list,), "world_model": (str,), "dataset": (str,),
            "sampling_constraints": (dict,), "replay_actions": (list,), "state_providers": (list,),
            "replay_transforms": (dict,)}, {"provenance": (dict,), "dataset_source_fps": (int, float),
                                         "evaluation_modes": (list,)})
        for key in ("id", "world_model", "dataset"):
            text(profile[key], location + "." + key)
        if profile["id"] in profile_ids:
            invalid(path, location + ".id", "Duplicate runtime profile id")
        profile_ids.add(profile["id"])
        for key in ("vlas", "replay_actions", "state_providers"):
            strings(profile[key], location + "." + key, nonempty=key == "vlas")
        if profile["dataset"] not in datasets or profile["world_model"] not in models["world_models"]:
            invalid(path, location, "Unknown runtime profile dataset/world model")
        for adapter in profile["vlas"]:
            if adapter not in models["vlas"]:
                invalid(path, location + ".vlas", f"Unknown VLA: {adapter}")
            identity = (adapter, profile["world_model"])
            if identity in identities:
                invalid(path, location, f"Duplicate runtime pair: {identity}")
            identities.add(identity)
        transforms = profile["replay_transforms"]
        if (set(transforms) != set(profile["replay_actions"]) or any(not isinstance(v, str) or v not in
                {"identity", "shuffle_rows", "zero_xyz_identity_rot6d"} for v in transforms.values())):
            invalid(path, location + ".replay_transforms", "Invalid replay action transform registration")
        for key, value in profile["sampling_constraints"].items():
            if key not in sampling or type(value) not in (str, bool, int, float) or (
                    type(value) in (int, float) and not math.isfinite(value)):
                invalid(path, location + ".sampling_constraints." + key, "Unknown sampling field or invalid scalar")
        if "dataset_source_fps" in profile:
            fps = profile["dataset_source_fps"]
            if not math.isfinite(fps) or fps <= 0:
                invalid(path, location + ".dataset_source_fps", "Expected finite positive FPS")
        if "evaluation_modes" in profile:
            strings(profile["evaluation_modes"], location + ".evaluation_modes", nonempty=True)
            if set(profile["evaluation_modes"]) - {"policy_loop", "pilworld"}:
                invalid(path, location + ".evaluation_modes", "Unknown policy evaluation mode")
        provenance = profile.get("provenance", {})
        fields(provenance, location + ".provenance", {}, {"git_sources": (list,), "files": (dict,),
            "wm_files": (dict,), "action_bridge_files": (list,), "source_files": (list,)})
        for key, value in provenance.items():
            if isinstance(value, list):
                strings(value, location + ".provenance." + key)
            else:
                for name, filename in value.items():
                    text(filename, location + ".provenance." + key + "." + name)
    return data
