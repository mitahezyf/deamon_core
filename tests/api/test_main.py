from fastapi.testclient import TestClient

from app.api.main import app


def test_lifespan_initializes_and_cleans_up_vox(mocker):
    vox_class = mocker.patch("app.api.main.DaemonVox", autospec=True)
    ears_class = mocker.patch("app.api.main.DaemonEars", autospec=True)
    stt_class = mocker.patch("app.api.main.DaemonStt", autospec=True)
    vox_instance = vox_class.return_value
    ears_instance = ears_class.return_value
    stt_instance = stt_class.return_value

    with TestClient(app) as client:
        assert client.app.state.vox is vox_instance
        assert client.app.state.ears is ears_instance
        assert client.app.state.stt is stt_instance
        vox_instance.load.assert_called_once()
        vox_instance.warmup.assert_called_once()
        ears_instance.load.assert_called_once()
        stt_instance.load.assert_called_once()

    assert not hasattr(app.state, "vox")
    assert not hasattr(app.state, "ears")
    assert not hasattr(app.state, "stt")
