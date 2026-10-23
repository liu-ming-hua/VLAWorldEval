"""Execution modes and PiL-World endpoint alignment, independent of models."""
import math


POLICY_MODES = frozenset({"policy_loop", "pilworld"})


def pilworld_alignment(native, world_action):
    """K stride-aligned future targets end at source action K*delta, not H_pi."""
    if native.semantics != "absolute_target":
        raise ValueError("pilworld requires decoded absolute targets; velocity/delta actions need a separately named baseline")
    for rate in (native.rate_hz, world_action.rate_hz):
        if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("pilworld action rates must be finite positive numbers")
    ratio = native.rate_hz / world_action.rate_hz
    if not math.isfinite(ratio) or ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-8):
        raise ValueError("pilworld requires an integer positive source-to-WM action stride")
    stride = int(round(ratio))
    steps = world_action.advance_steps * stride
    if steps > native.horizon:
        raise ValueError("pilworld terminal action exceeds the native horizon; no implicit tail padding")
    return {"action_stride": stride, "executed_action_steps": steps,
            "native_action_index": steps - 1, "wm_action_index": world_action.horizon - 1,
            "native_horizon": native.horizon}
