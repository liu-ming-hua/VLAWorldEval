"""README shell redirection must yield one JSON document, never progress text."""
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

from cli.commands import execution, studies, verification
from evaluation.execution.reliability import RunInterrupted


@pytest.mark.parametrize("handler,target", [
    (execution.run, "evaluation.execution.executor.execute_suite"),
    (execution.evaluate, "evaluation.execution.workflow.evaluate_suite"),
    (execution.replay, "evaluation.execution.native_replay.run_native_replay"),
    (execution.policy_check, "evaluation.execution.policy_check.run_policy_check"),
    (verification.accept, "evaluation.validation.acceptance.run_acceptance"),
    (studies.droid_study, "evaluation.execution.droid_study.run_study"),
])
def test_execution_reports_are_json_and_plugin_progress_is_stderr(monkeypatch, capsys, handler, target):
    expected = {"ok": True, "status": "passed", "execution_performed": False}

    def operation(*args, **kwargs):
        print("fixture dataset progress, not scientific evidence")
        return expected

    monkeypatch.setattr(target, operation)
    args = SimpleNamespace(config=Path("suite.json"), experiment=[], seed=[], admit_only=False,
        execute=True, input=Path("input.npz"), output=Path("output"), matrix=Path("matrix.json"),
        pair=[], level="smoke", operation="run", directory=Path("study"), phase="all", watch=False,
        model="dreamzero_droid", dataset=Path("dataset"), episode=350, start=15,
        gpus=None, memory_gib=None, attention_backend=None, runtime_check=False)
    assert handler(args, Path.cwd()) == 0
    output = capsys.readouterr()
    assert json.loads(output.out) == expected
    assert "fixture dataset progress" in output.err
    print("stdout restored")
    assert capsys.readouterr().out == "stdout restored\n"


def test_study_cancellation_has_signal_exit_code_and_no_partial_json(monkeypatch, capsys):
    def interrupted(*args, **kwargs):
        raise RunInterrupted(signal.SIGTERM)
    monkeypatch.setattr("evaluation.execution.droid_study.run_study", interrupted)
    args = SimpleNamespace(operation="run", directory=Path("study"), phase="all", watch=True)
    assert studies.droid_study(args, Path.cwd()) == 143
    output = capsys.readouterr()
    assert not output.out
    assert "SIGTERM" in output.err
