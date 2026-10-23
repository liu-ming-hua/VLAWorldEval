"""Evidence-bound local task annotations and matched real/model rankings.

No cloud calls, automatic success guesses, or fabricated missing outcomes.
Ranking concordance is descriptive, not a statistical validation claim.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from itertools import combinations


def load_annotations(path: Path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value.get("schema_version") != 1 or not isinstance(value.get("records"), list):
        raise ValueError("Annotations require schema_version=1 and records list")
    seen = set()
    for record in value["records"]:
        if not isinstance(record, dict):
            raise ValueError("Each annotation must be an object")
        for key in ("policy", "task", "episode", "seed", "domain", "evaluator", "rubric", "evidence", "evidence_sha256"):
            if key not in record or not isinstance(record[key], str) or not record[key].strip():
                raise ValueError(f"Annotation missing nonempty string {key}")
        if record["domain"] not in {"real", "world_model"}:
            raise ValueError("domain must be real or world_model")
        identity = tuple(record[k] for k in ("policy", "task", "episode", "seed", "domain"))
        if identity in seen:
            raise ValueError(f"Duplicate outcome: {identity}")
        seen.add(identity)
        score = record.get("score")
        if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))
                                  or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("score must be null or finite in [0,1]")
        if score is None and not record.get("unscored_reason"):
            raise ValueError("Unscored outcomes require unscored_reason")
        evidence = (path.parent / record["evidence"]).resolve()
        if not evidence.is_file():
            raise ValueError(f"Missing evidence: {evidence}")
        digest = _digest(evidence)
        if digest != record["evidence_sha256"]:
            raise ValueError(f"Evidence hash mismatch: {evidence}")
    return value["records"]


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_scores(records):
    groups = {}
    for record in records:
        key = (record["policy"], record["domain"], record["rubric"])
        group = groups.setdefault(key, {"policy": key[0], "domain": key[1], "rubric": key[2], "scores": [], "unscored": 0})
        if record.get("score") is None:
            group["unscored"] += 1
        else:
            group["scores"].append(record["score"])
    summaries = []
    for key in sorted(groups):
        group = groups[key]
        scores = group.pop("scores")
        summaries.append({**group, "scored": len(scores),
                          "mean_score": sum(scores) / len(scores) if scores else None})
    # Only compare the same task/episode/seed/rubric with both outcomes scored.
    cases = {}
    for record in records:
        key = tuple(record[k] for k in ("task", "episode", "seed", "rubric"))
        cases.setdefault(key, {}).setdefault(record["policy"], {})[record["domain"]] = record.get("score")
    policies = sorted({r["policy"] for r in records})
    comparisons = []
    for left, right in combinations(policies, 2):
        matched = [case for case in cases.values() if all(
            p in case and all(case[p].get(d) is not None for d in ("real", "world_model"))
            for p in (left, right))]
        if not matched:
            continue
        deltas = {domain: sum(case[left][domain] - case[right][domain] for case in matched) / len(matched)
                  for domain in ("real", "world_model")}
        tied = any(abs(delta) < 1e-12 for delta in deltas.values())
        comparisons.append({"policies": [left, right], "matched_cases": len(matched), "mean_deltas": deltas,
                            "concordant": None if tied else deltas["real"] * deltas["world_model"] > 0})
    eligible = [item for item in comparisons if item["concordant"] is not None]
    return {"schema_version": 1, "groups": summaries, "matched_policy_pairs": comparisons,
            "pairwise_concordance": (sum(item["concordant"] for item in eligible) / len(eligible)) if eligible else None,
            "scientific_validation_claimed": False, "automatic_task_scoring": False}
