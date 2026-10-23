"""Public CLI: parse, dispatch and report structured failures."""

import json
import sys
import subprocess
from pathlib import Path

from vlaworldeval.paths import project_root, require_asset_project
from cli.parser import build_parser
from cli.commands import COMMANDS, STANDALONE_COMMANDS

try:
    ROOT = project_root()
    _ROOT_ERROR = None
except ValueError as exc:
    ROOT = Path(__file__).resolve().parents[1]
    _ROOT_ERROR = str(exc)


def _main(argv=None):
    args = build_parser(ROOT).parse_args(argv)
    if args.command not in STANDALONE_COMMANDS:
        if _ROOT_ERROR is not None:
            raise ValueError(_ROOT_ERROR)
        require_asset_project(ROOT)
    return COMMANDS[args.command](args, ROOT)


def main(argv: list[str] | None = None) -> int:
    from evaluation.planning.configuration import ConfigurationError
    from evaluation.execution.admission import ResourceAdmissionError
    from evaluation.execution.reliability import WorkerError

    debug = "--debug" in (sys.argv[1:] if argv is None else argv)
    try:
        return _main(argv)
    except ConfigurationError as exc:
        if debug:
            raise
        print(json.dumps({"error": exc.details}, ensure_ascii=False), file=sys.stderr)
        return 2
    except WorkerError as exc:
        if debug:
            raise
        print(json.dumps({"error": exc.details}, ensure_ascii=False), file=sys.stderr)
        return 1
    except (ValueError, OSError, ResourceAdmissionError, subprocess.SubprocessError) as exc:
        if debug:
            raise
        print(
            json.dumps(
                {
                    "error": {
                        "code": "admission_error"
                        if isinstance(exc, ResourceAdmissionError)
                        else "input_error",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
