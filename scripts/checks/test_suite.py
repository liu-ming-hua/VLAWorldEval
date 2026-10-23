"""Inventory test ownership and declared costs, not production-line coverage."""
import argparse
import ast
from collections import Counter, defaultdict
import json
from pathlib import Path


GROUPS = {"core", "models", "engineering", "integration"}
PRODUCTION = {"adapters", "evaluation", "vlaworldeval", "cli", "scripts"}


def inventory(root):
    root = Path(root).resolve()
    records, bodies = [], defaultdict(list)
    for path in sorted((root / "tests").rglob("test_*.py")):
        relative = path.relative_to(root)
        if len(relative.parts) != 3 or relative.parts[1] not in GROUPS:
            raise ValueError(f"Test modules need one functional group: {relative}")
        tree = ast.parse(path.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")]
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        imports = sorted(name for name in imported if name.split(".")[0] in PRODUCTION)
        process = any(isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets)
                      and isinstance(node.value, ast.Attribute) and node.value.attr == "process"
                      for node in tree.body)
        tier = "integration" if relative.parts[1] == "integration" else "process" if process else "unit"
        records.append({"module": relative.as_posix(), "group": relative.parts[1], "tier": tier,
                        "test_functions": len(functions), "production_imports": imports})
        for function in functions:
            body = ast.dump(ast.Module(body=function.body, type_ignores=[]), include_attributes=False)
            bodies[body].append(f"{relative.as_posix()}::{function.name}")
    return {"schema_version": 1, "modules": len(records),
            "test_functions": sum(record["test_functions"] for record in records),
            "groups": dict(Counter(record["group"] for record in records)),
            "tiers": dict(Counter(record["tier"] for record in records)),
            "exact_duplicate_function_bodies": [names for names in bodies.values() if len(names) > 1],
            "records": records,
            "scope": "Static ownership/import map; not measured executed-line or branch coverage"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--json", action="store_true", help="Include every module and its imported production targets")
    args = parser.parse_args()
    report = inventory(args.root)
    if not args.json:
        report.pop("records")
    print(json.dumps(report, indent=2, ensure_ascii=False))
