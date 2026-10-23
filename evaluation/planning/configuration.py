"""Suite shape validation with file/field diagnostics; no model imports."""
import json
from pathlib import Path


class ConfigurationError(ValueError):
    def __init__(self, path, issues):
        self.details = {"code": "invalid_configuration", "file": str(path), "issues": issues}
        super().__init__("; ".join(f"{issue['field']}: {issue['message']}" for issue in issues))


def invalid(path, field, message):
    raise ConfigurationError(path, [{"field": field, "message": message}])


def load_suite(path):
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        invalid(path, "$", f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}")
    except OSError as exc:
        invalid(path, "$", str(exc))
    issues = []

    def object_fields(obj, field, required, optional=None):
        if not isinstance(obj, dict):
            issues.append({"field": field or "$", "message": "Expected an object"})
            return False
        for key in obj.keys() - required.keys() - (optional or {}).keys():
            issues.append({"field": f"{field}.{key}" if field else key, "message": "Unknown field (check spelling)"})
        for key, types in {**required, **(optional or {})}.items():
            location = f"{field}.{key}" if field else key
            if key not in obj:
                if key in required:
                    issues.append({"field": location, "message": "Required field is missing"})
                continue
            if type(obj[key]) not in types:
                issues.append({"field": location, "message": f"Expected {'/'.join(t.__name__ for t in types)}, got {type(obj[key]).__name__}"})
        return True

    text, number, integer, array, boolean = (str,), (int, float), (int,), (list,), (bool,)
    if object_fields(value, "", {"schema_version": integer, "suite_id": text, "pair_manifest": text,
                                  "dataset": (dict,), "runtime": (dict,), "sampling": (dict,),
                                  "seeds": array, "output_root": text, "experiments": array}, {"metadata": (dict,)}):
        object_fields(value.get("dataset"), "dataset", {"root": text, "episodes": array, "start_frame": integer,
                                                       "fps": number, "tail_policy": text},
                      {"loader": text, "instruction": text})
        object_fields(value.get("runtime"), "runtime", {
            "vla_environment": text, "world_model_environment": text, "vla_gpus": array,
            "world_model_gpus": array, "execution": text, "scheduling": text,
            "max_process_memory_gib_per_gpu": number, "world_model_checkpoint": text},
            {"vla_memory_gib": number, "worker_options": (dict,), "vla_worker_options": (dict,),
             "world_model_worker_options": (dict,)})
        object_fields(value.get("sampling"), "sampling", {
            "num_inference_steps": integer, "guidance": number, "shift": number,
            "image_size": integer, "resolution": text, "diffusion_cache": boolean, "torch_compile": boolean})
        if isinstance(value.get("experiments"), list):
            for i, experiment in enumerate(value["experiments"]):
                object_fields(experiment, f"experiments[{i}]", {"id": text, "mode": text,
                              "action_source": text, "state_provider": text, "chunks": integer},
                              {"seeds": array, "episodes": array})
    if issues:
        raise ConfigurationError(path, issues)
    return value
