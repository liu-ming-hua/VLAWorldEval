# Command-line interface

One public entry point: `python -m cli` or installed `vlaworldeval`.

- `parser.py`: arguments only; no model imports.
- `commands/`: grouped handlers.
- `output.py`: structured, finite JSON output.
- `main.py`: dispatch and consistent errors.

Use `--help` for current flags. Inference requires explicit execution commands;
planning and inspection do not load models.

[Quick start](../docs/en/quickstart.md) · [Workflows](../docs/en/workflows.md) · [中文手册](../docs/zh/README.md)
