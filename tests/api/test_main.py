from fastapi.testclient import TestClient

from app.api.main import app


def test_lifespan_initializes_and_cleans_up_vox(mocker):
    vox_class = mocker.patch("app.api.main.DaemonVox", autospec=True)
    vox_instance = vox_class.return_value

    with TestClient(app) as client:
        assert client.app.state.vox is vox_instance
        vox_instance.load.assert_called_once()
        vox_instance.warmup.assert_called_once()

    assert not hasattr(app.state, "vox")
