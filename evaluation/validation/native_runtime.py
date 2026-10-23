"""Import/ABI checks in the model interpreter, with GPUs hidden and HF offline."""
import json
import os
from pathlib import Path
import subprocess

from vlaworldeval.cache import runtime_cache_environment


_PROBE = r'''
import contextlib, hashlib, importlib, importlib.util, json, pathlib, sys
request = json.loads(sys.argv[1])
report = {"ok": False, "scope": "CPU_imports_not_model_acceptance",
          "model_inference_performed": False, "python": sys.executable,
          "modules": {}, "errors": []}
with contextlib.redirect_stdout(sys.stderr):
    try:
        spec = importlib.util.find_spec(request["package"])
        origin = None if spec is None else spec.origin
        report["package_origin"] = origin
        if not origin or not pathlib.Path(origin).resolve().is_relative_to(pathlib.Path(request["source"]).resolve()):
            raise ImportError("Model package does not resolve inside the selected upstream source")
        package_root = pathlib.Path(origin).resolve().parent
        hashes = {str(path.relative_to(package_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(package_root.rglob("*.py"))}
        report["package_python_sources"] = {
            "root": str(package_root), "files": len(hashes),
            "sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}
        for name in request["imports"]:
            try:
                module = importlib.import_module(name)
                path = getattr(module, "__file__", None)
                version = getattr(module, "__version__", None)
                report["modules"][name] = {"origin": path, "version": None if version is None else str(version)}
            except Exception as exc:
                report["errors"].append({"module": name, "type": type(exc).__name__, "message": str(exc)})
        torch = sys.modules.get("torch")
        report["cuda_initialized"] = bool(torch is not None and torch.cuda.is_initialized())
        if report["cuda_initialized"]:
            raise RuntimeError("CPU runtime probe unexpectedly initialized CUDA")
    except Exception as exc:
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
report["ok"] = not report["errors"]
print(json.dumps(report))
'''


def probe_native_runtime(python, *, root, source, package, imports, timeout=45):
    """No downloads, assets, model constructors, local PYTHONPATH or CUDA context.

    Imported extensions may load CPU-side shared libraries. Success only proves
    these imports; it does not prove GPU kernels, model loading or full readiness.
    """
    root, source = Path(root).resolve(), Path(source).resolve()
    env = os.environ.copy()
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "PYTHONPATH"):
        env.pop(name, None)
    env.update(runtime_cache_environment(root))
    env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               NO_ALBUMENTATIONS_UPDATE="1", USE_TF="0", COSMOS_TRAINING="0", COSMOS_INTERNAL="0")
    request = {"source": str(source), "package": package, "imports": list(imports)}
    try:
        process = subprocess.run([str(python), "-I", "-c", _PROBE, json.dumps(request)],
                                 cwd=root, env=env, capture_output=True, text=True, timeout=timeout)
        report = json.loads(process.stdout)
        if not isinstance(report, dict) or type(report.get("ok")) is not bool:
            raise ValueError("Invalid runtime probe report")
        report.update(returncode=process.returncode, stderr=process.stderr)
        report["ok"] = report["ok"] and process.returncode == 0
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        report = {"ok": False, "scope": "CPU_imports_not_model_acceptance",
                  "model_inference_performed": False,
                  "errors": [{"type": type(exc).__name__, "message": str(exc)}]}
    report.update(request=request, gpu_visibility="disabled", proxies_removed=True)
    return report
