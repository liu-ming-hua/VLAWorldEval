import hashlib
import json

import pytest

from evaluation.reporting.task_scores import load_annotations, summarize_scores


def annotations(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"fixture": true}')
    records = [
        {
            "policy": policy,
            "task": "pick",
            "episode": "e1",
            "seed": "0",
            "domain": domain,
            "evaluator": "human",
            "rubric": "pick-v1",
            "evidence": evidence.name,
            "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
            "score": score,
        }
        for policy, domain, score in [
            ("a", "real", 1),
            ("a", "world_model", 1),
            ("b", "real", 0),
            ("b", "world_model", 0),
        ]
    ]
    path = tmp_path / "scores.json"
    path.write_text(json.dumps({"schema_version": 1, "records": records}))
    return path, records


def test_local_scores_are_evidence_bound_and_matched(tmp_path):
    path, records = annotations(tmp_path)
    report = summarize_scores(load_annotations(path))
    assert report["pairwise_concordance"] == 1
    assert not report["scientific_validation_claimed"]
    records[-1]["episode"] = "different"
    assert summarize_scores(records)["pairwise_concordance"] is None
    records[-1]["score"] = None
    assert summarize_scores(records)["groups"][-1]["mean_score"] is None
    (tmp_path / "evidence.json").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_annotations(path)


@pytest.mark.parametrize("score", [True, -1, 2, float("nan"), None])
def test_invalid_or_unexplained_scores_rejected(tmp_path, score):
    path, records = annotations(tmp_path)
    records[0]["score"] = score
    path.write_text(json.dumps({"schema_version": 1, "records": records}))
    with pytest.raises(ValueError):
        load_annotations(path)


@pytest.mark.parametrize(
    "payload",
    [[], None, {"schema_version": True, "records": []}, {"schema_version": 1, "records": [[]]}],
)
def test_malformed_annotations_raise_input_errors(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_annotations(path)
