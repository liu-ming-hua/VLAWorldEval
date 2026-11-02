"""Own a local model server process group and always release it after replay."""
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request


@contextmanager
def local_server(command, *, cwd, environment, log_path, health_url, timeout=600):
    """Yield lifecycle evidence only after readiness; never reuse an existing server."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    record = {'command': list(map(str, command)), 'ready': False, 'pid': None,
              'returncode': None, 'shutdown_escalated': False}
    with Path(log_path).open('w') as log:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        record['pid'] = process.pid
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'Model server exited before readiness; see {log_path}')
                try:
                    with opener.open(health_url, timeout=1) as response:
                        if response.status == 200:
                            record['ready'] = True
                            break
                except OSError:
                    pass
                time.sleep(0.5)
            if not record['ready']:
                raise TimeoutError(f'Model server readiness timed out; see {log_path}')
            yield record
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    record['shutdown_escalated'] = True
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            record['returncode'] = process.returncode
