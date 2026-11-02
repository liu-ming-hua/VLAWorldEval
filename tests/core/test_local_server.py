"""Owned server cleanup on normal exit and caller failures, without GPU assets."""
import os
import socket
import sys

import pytest

from evaluation.execution.local_server import local_server

pytestmark = pytest.mark.process


@pytest.mark.parametrize('failure', [False, True])
def test_owned_server_cleanup(tmp_path, failure):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    record = None
    try:
        with local_server([sys.executable, '-m', 'http.server', str(port), '--bind', '127.0.0.1'],
                          cwd=tmp_path, environment=dict(os.environ), log_path=tmp_path / 'server.log',
                          health_url=f'http://127.0.0.1:{port}/', timeout=10) as record:
            assert record['ready']
            if failure:
                raise ValueError('caller failure')
    except ValueError as exc:
        assert failure and str(exc) == 'caller failure'
    assert record['returncode'] is not None
    assert not record['shutdown_escalated']


def test_server_start_failure_is_visible(tmp_path):
    with pytest.raises(RuntimeError, match='exited before readiness'):
        with local_server([sys.executable, '-c', 'raise SystemExit(7)'], cwd=tmp_path,
                          environment=dict(os.environ), log_path=tmp_path / 'failed.log',
                          health_url='http://127.0.0.1:1/', timeout=5):
            pytest.fail('Failed server must never admit inference')
