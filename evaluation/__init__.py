"""Public evaluation API with dependency-light lazy imports."""

from importlib import import_module

from .contracts import ActionChunk, ObservationBundle, RobotState, WorldModelOutput

__all__ = [
    "PolicyWorldModelEvaluator",
    "RolloutTrace",
    "ActionChunk",
    "ObservationBundle",
    "RobotState",
    "WorldModelOutput",
    "PairManifest",
    "load_pair_manifest",
    "aggregate_model_results",
    "discover_trials",
    "encode_video",
    "load_tasks",
    "print_results_table",
    "score_video_with_vlm",
]

_LAZY_IMPORTS = {
    "PolicyWorldModelEvaluator": ("evaluation.evaluator", "PolicyWorldModelEvaluator"),
    "RolloutTrace": ("evaluation.evaluator", "RolloutTrace"),
    "PairManifest": ("evaluation.manifests", "PairManifest"),
    "load_pair_manifest": ("evaluation.manifests", "load_pair_manifest"),
    "aggregate_model_results": ("evaluation.reporting.scoring", "aggregate_model_results"),
    "encode_video": ("evaluation.reporting.scoring", "encode_video"),
    "print_results_table": ("evaluation.reporting.scoring", "print_results_table"),
    "score_video_with_vlm": ("evaluation.reporting.scoring", "score_video_with_vlm"),
    "discover_trials": ("evaluation.datasets.trials", "discover_trials"),
    "load_tasks": ("evaluation.datasets.trials", "load_tasks"),
}


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    module_name, attribute_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
