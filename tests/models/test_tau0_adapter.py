"""Simulator transport, RGB conversion and autoregressive feedback contracts."""
import numpy as np
import pytest

from adapters.world_models.tau0_wm_simulator import Tau0WMSimulatorAdapter


class Client:
    def __init__(self):
        self.requests = []
        self.fail = False

    def infer(self, payload):
        self.requests.append(payload)
        if self.fail:
            raise RuntimeError('fixture transport failure')
        return {'results': [np.full((2, 3, 4, 6), 0.5, np.float32), np.zeros(7)]}


def adapter(tmp_path):
    model = Tau0WMSimulatorAdapter(repo_path=tmp_path, image_size=(4, 6))
    model._client = Client()
    model.reset([np.zeros((4, 6, 3), np.uint8), np.full((4, 6, 3), 100, np.uint8)])
    return model


def test_feedback_updates_from_generated_frame_and_resets_once(tmp_path):
    model = adapter(tmp_path)
    action = np.ones((7, 20), np.float32)
    first = model.step_output(action)
    assert first.metadata['output_kind'] == 'terminal_frame_only'
    assert not first.metadata['full_video_available']
    assert first.views['static'].shape == (1, 4, 6, 3)
    assert np.all(first.views['static'] == 191)  # fixed [-1,1], not range guessing
    model.step(action)
    requests = model._client.requests
    assert requests[0]['prompt'].startswith('<reset>')
    assert not requests[1]['prompt'].startswith('<reset>')
    np.testing.assert_allclose(requests[1]['obs'], 191 / 127.5 - 1, atol=1e-6)
    assert requests[0]['execution_step'] == 7
    np.testing.assert_array_equal(requests[0]['actions'], action)


@pytest.mark.parametrize('shape', [(33, 7), (34, 20), (0, 20)])
def test_invalid_action_never_contacts_server(tmp_path, shape):
    model = adapter(tmp_path)
    with pytest.raises(ValueError):
        model.step(np.zeros(shape, np.float32))
    assert not model._client.requests


def test_reset_failure_can_be_retried(tmp_path):
    model = adapter(tmp_path)
    model._client.fail = True
    with pytest.raises(RuntimeError):
        model.step(np.ones((1, 20), np.float32))
    assert model._needs_reset
    model._client.fail = False
    model.step(np.ones((1, 20), np.float32))
    assert model._client.requests[-1]['prompt'].startswith('<reset>')


class VideoClient(Client):
    def infer(self, payload):
        result = super().infer(payload)
        video = np.zeros((2, 3, 9, 4, 6), np.float32)
        video[:, :, 1:8] = -0.5
        video[:, :, -1] = 0.5
        result['results'].append({
            'protocol': 'vlaworldeval.tau0.video.v1', 'video': video,
            'layout': 'VCTHW', 'range': '[-1,1]', 'includes_context_frame': True,
        })
        return result


def test_full_video_preserves_future_frames_and_terminal_feedback(tmp_path):
    model = Tau0WMSimulatorAdapter(repo_path=tmp_path, image_size=(4, 6), return_full_video=True)
    model._client = VideoClient()
    model.reset(np.zeros((2, 4, 6, 3), np.uint8))
    result = model.step_output(np.zeros((33, 20), np.float32))
    assert result.views['static'].shape == (8, 4, 6, 3)
    assert np.all(result.views['static'][:-1] == 64)
    assert np.all(result.views['static'][-1] == 191)
    assert result.metadata['full_video_available']
    assert result.metadata['context_frame_removed']
    terminal = model.step(np.zeros((33, 20), np.float32))
    assert np.all(terminal == 191)
    np.testing.assert_allclose(model._client.requests[-1]['obs'], 191 / 127.5 - 1, atol=1e-6)
    assert model._client.requests[-1]['return_full_video']


def test_full_video_rejects_legacy_server_without_advancing_feedback(tmp_path):
    model = adapter(tmp_path)
    model.return_full_video = True
    with pytest.raises(ValueError, match='lacks the full-video extension'):
        model.step(np.zeros((33, 20), np.float32))
    assert model._needs_reset
    assert model.last_video is None


@pytest.mark.parametrize('fault', ['protocol', 'terminal', 'nan', 'shape'])
def test_malformed_full_video_rejected(tmp_path, fault):
    model = adapter(tmp_path)
    model.return_full_video = True
    client = VideoClient()
    original = client.infer

    def bad(payload):
        response = original(payload)
        data = response['results'][2]
        if fault == 'protocol':
            data['layout'] = 'THWVC'
        elif fault == 'terminal':
            data['video'][:, :, -1] = 0
        elif fault == 'nan':
            data['video'][:, :, 3] = np.nan
        else:
            data['video'] = data['video'][0]
        return response

    client.infer = bad
    model._client = client
    with pytest.raises(ValueError):
        model.step(np.zeros((33, 20), np.float32))
    assert model.last_video is None
