"""Consistent, machine-readable command output."""

import json


def print_json(value):
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
