"""Dispatch table for public commands; kept free of model imports."""

from . import workspace, project, execution, reports, verification, studies

COMMANDS = {
    "cleanup-results": workspace.cleanup_results,
    "init": workspace.init,
    "verify-assets": workspace.verify_assets,
    "configure": workspace.configure,
    "demo": workspace.demo,
    "models": project.models,
    "environments": project.environments,
    "validate": project.validate,
    "runtimes": project.runtimes,
    "capabilities": project.capabilities,
    "plan": project.plan,
    "doctor": project.doctor,
    "run": execution.run,
    "policy-check": execution.policy_check,
    "evaluate": execution.evaluate,
    "replay": execution.replay,
    "native-study": studies.native_study,
    "replay-report": reports.replay_report,
    "motion-eval": reports.motion_eval,
    "prepare-replay": execution.prepare_replay,
    "results": reports.results,
    "inspect": reports.inspect,
    "scores": reports.scores,
    "regression-report": reports.regression_report,
    "accept": verification.accept,
    "env-audit": verification.env_audit,
    "engineering-accept": verification.engineering_accept,
    "droid-study": studies.droid_study,
}

STANDALONE_COMMANDS = {"init", "verify-assets", "scores", "demo", "inspect", "results"}
