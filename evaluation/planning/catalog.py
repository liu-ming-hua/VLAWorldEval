"""Validated model descriptions; declarations never substitute for execution."""
import json

from evaluation.planning.configuration import invalid


def load_model_catalog(root):
    path = root / "configs/model_catalog.json"
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        invalid(path, "$", str(exc))
    if not isinstance(catalog, dict):
        invalid(path, "$", "Expected a model catalog object")
    if type(catalog.get("schema_version")) is not int or catalog["schema_version"] != 1:
        invalid(path, "schema_version", "Expected integer schema version 1")
    if "description" in catalog and not isinstance(catalog["description"], str):
        invalid(path, "description", "Expected a string")
    for group in ("vlas", "world_models"):
        entries = catalog.get(group)
        if not isinstance(entries, dict) or not entries:
            invalid(path, group, "Expected a nonempty model mapping")
        for name, entry in entries.items():
            location = f"{group}.{name}"
            if not name.strip() or not isinstance(entry, dict):
                invalid(path, location, "Expected a named model object")
            texts = ["adapter", "environment", "training_domain", "source", "status"]
            texts += ["native_output"] if group == "vlas" else ["action_bridge"]
            for field in texts:
                value = entry.get(field)
                if not isinstance(value, str) or not value.strip():
                    invalid(path, f"{location}.{field}", "Required nonempty string")
            if not entry["adapter"].isidentifier():
                invalid(path, location + ".adapter", "Expected an adapter class name")
            for field in ("input",) if group == "vlas" else ("input", "output"):
                if not isinstance(entry.get(field), dict) or not entry[field]:
                    invalid(path, f"{location}.{field}", "Required nonempty interface description object")
            for field in ("adapter_output", "action_semantics"):
                if field in entry and (not isinstance(entry[field], str) or not entry[field].strip()):
                    invalid(path, f"{location}.{field}", "Expected a nonempty string")
            if "integration" in entry:
                integration = entry["integration"]
                if not isinstance(integration, dict) or set(integration) != {"level", "scope", "reason", "supported_pair_ids"}:
                    invalid(path, location + ".integration", "Require level, scope, reason and supported_pair_ids")
                if integration["level"] not in ("supported", "native_replay", "experimental", "incompatible"):
                    invalid(path, location + ".integration.level", "Unknown declared engineering tier")
                for field in ("scope", "reason"):
                    if not isinstance(integration[field], str) or not integration[field].strip():
                        invalid(path, location + ".integration." + field, "Require an explicit nonempty boundary")
                pairs = integration["supported_pair_ids"]
                if (not isinstance(pairs, list) or any(not isinstance(v, str) or not v for v in pairs)
                        or len(set(pairs)) != len(pairs)):
                    invalid(path, location + ".integration.supported_pair_ids", "Require unique string pair IDs")
                if bool(pairs) != (integration["level"] == "supported"):
                    invalid(path, location + ".integration.supported_pair_ids", "Only supported tiers declare supported pair IDs; they must name at least one")
    return catalog
