import json

import pytest

from evaluation.planning.native_subset import (
    audit_reference_subset,
    freeze_reference_subset,
    select_cases,
)


def cases(count):
    return [{"id": f"case{i}", "prepared": f"datasets/case{i}"} for i in range(count)]


def test_selection_is_order_independent_and_has_no_replacement():
    population = cases(1000)
    selected = select_cases(population)
    assert len(selected) == len({c["id"] for c in selected}) == 400
    assert selected == select_cases(list(reversed(population)))
    assert selected != select_cases(population, seed=43)


@pytest.mark.parametrize("limit,seed", [(0, 42), (True, 42), (400, True)])
def test_invalid_budget_rejected(limit, seed):
    with pytest.raises(ValueError):
        select_cases(cases(2), limit, seed)


def test_freeze_preserves_protocol_and_shortfall_without_mutating_source(tmp_path):
    source = tmp_path / "population.json"
    plan = {
        "schema_version": 1,
        "scope": "official",
        "cases": cases(5),
        "unavailable": {"gated": "unavailable"},
    }
    source.write_text(json.dumps(plan))
    before = source.read_bytes()
    result = freeze_reference_subset(tmp_path, source, tmp_path / "budget")
    assert result["selected_cases"] == 5 and result["shortfall"] == 395
    assert not result["full_test_set_claim"]
    frozen = json.loads((tmp_path / "budget/plan.json").read_text())
    assert frozen["unavailable"] == plan["unavailable"]
    assert source.read_bytes() == before
    with pytest.raises(FileExistsError):
        freeze_reference_subset(tmp_path, source, tmp_path / "budget")


def frozen_budget(root, count=5, limit=2):
    source = root / "population.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "official released examples",
                "cases": cases(count),
                "unavailable": {"full_test": "unknown"},
            }
        )
    )
    freeze_reference_subset(root, source, root / "budget", limit=limit)
    path = root / "budget/plan.json"
    return path, json.loads(path.read_text())


@pytest.mark.parametrize("count,limit", [(5, 2), (2, 5)])
def test_budget_audit_reproduces_membership_not_paper_test_claim(tmp_path, count, limit):
    path, plan = frozen_budget(tmp_path, count, limit)
    value = audit_reference_subset(tmp_path, path, plan)
    assert value["selected_fraction_of_source_cases"] == min(count, limit) / count
    assert value["source_membership_verified"]
    assert not value["source_is_complete_official_test_set_verified"]
    assert not value["full_test_set_claim"]


@pytest.mark.parametrize(
    "target", ["source", "ids", "count", "seed", "method", "full_claim", "plan"]
)
def test_budget_audit_rejects_changed_evidence(tmp_path, target):
    path, plan = frozen_budget(tmp_path)
    selection = path.parent / "selection.json"
    value = json.loads(selection.read_text())
    if target == "source":
        (tmp_path / "population.json").write_text("{}")
    elif target == "plan":
        plan["cases"][0]["prepared"] = "datasets/substitute"
    else:
        key, changed = {
            "ids": ("selected_ids", ["case0"]),
            "count": ("population_cases", 2),
            "seed": ("sampling_seed", True),
            "method": ("sampling_method", "quality_sorted"),
            "full_claim": ("full_test_set_claim", True),
        }[target]
        value[key] = changed
        selection.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        audit_reference_subset(tmp_path, path, plan)


def test_without_selection_audit_does_not_invent_population(tmp_path):
    assert audit_reference_subset(tmp_path, tmp_path / "plan.json", {"cases": cases(2)}) is None
