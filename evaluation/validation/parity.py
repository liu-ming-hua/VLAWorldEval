"""Compare captured native/adapter traces without importing model dependencies."""
import json
from pathlib import Path

import numpy as np

from evaluation.execution.transport import read_packet
from evaluation.reporting.support import digest


def compare_values(left, right, *, atol=0, rtol=0, path='root'):
    """Exact structure/dtype checks; declared tolerances apply only to array values."""
    errors, metrics = [], {}
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            return [f'{path}: array/type mismatch'], metrics
        if left.shape != right.shape or left.dtype != right.dtype:
            return [f'{path}: shape/dtype mismatch'], metrics
        if left.dtype.kind not in 'biuf' or not np.isfinite(left).all() or not np.isfinite(right).all() or left.size == 0:
            return [f'{path}: empty, nonnumeric or nonfinite array'], metrics
        delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
        metrics[path] = {'shape': list(left.shape), 'dtype': str(left.dtype),
                         'max_abs': float(delta.max()), 'mean_abs': float(delta.mean())}
        if not np.allclose(left, right, atol=atol, rtol=rtol):
            errors.append(f'{path}: values exceed tolerance')
    elif isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return [f'{path}: dictionary keys differ'], metrics
        for key in sorted(left):
            e, m = compare_values(left[key], right[key], atol=atol, rtol=rtol, path=f'{path}.{key}')
            errors.extend(e)
            metrics.update(m)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f'{path}: list lengths differ'], metrics
        for i, (a, b) in enumerate(zip(left, right)):
            e, m = compare_values(a, b, atol=atol, rtol=rtol, path=f'{path}[{i}]')
            errors.extend(e)
            metrics.update(m)
    elif type(left) is not type(right) or left != right:
        errors.append(f'{path}: scalar/type mismatch')
    return errors, metrics


def compare_traces(native_path, adapter_path, policy_path, output):
    policy_path, output = Path(policy_path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    policy = json.loads(policy_path.read_text())
    atol, rtol = policy['atol'], policy['rtol']
    if any(type(v) not in (int, float) or not np.isfinite(v) or v < 0 for v in (atol, rtol)):
        raise ValueError('Finite nonnegative tolerances required')
    native, adapter = read_packet(Path(native_path)), read_packet(Path(adapter_path))
    errors = []
    for trace, route in [(native, 'native'), (adapter, 'adapter')]:
        if (trace.get('schema_version') != 1 or trace.get('route') != route
                or trace.get('profile') != policy['profile'] or trace.get('protocol') != policy['protocol'] or not trace.get('inputs')
                or not trace.get('outputs') or not trace.get('provenance')):
            raise ValueError('Trace identity, inputs, outputs and provenance required')
    for key in ('protocol', 'seed', 'asset_lock_sha256', 'native_revision'):
        if key not in native or key not in adapter or native[key] != adapter[key]:
            errors.append(f'{key}: missing or mismatched binding')
    input_errors, input_metrics = compare_values(native['inputs'], adapter['inputs'], path='inputs')
    output_errors, output_metrics = compare_values(native['outputs'], adapter['outputs'], atol=atol, rtol=rtol, path='outputs')
    errors += input_errors + output_errors
    report = {'schema_version': 1, 'kind': 'native_adapter_parity', 'profile': policy['profile'],
              'status': 'failed' if errors else 'passed', 'errors': errors,
              'policy_sha256': digest(policy_path), 'policy': policy,
              'trace_sha256': {'native': digest(native_path), 'adapter': digest(adapter_path)},
              'input_metrics': input_metrics, 'output_metrics': output_metrics,
              'provenance': {'native': native['provenance'], 'adapter': adapter['provenance']},
              'scope': 'Captured inputs and outputs in this runtime/profile only; not physical reliability or all model versions',
              'physical_reliability_verified': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report
