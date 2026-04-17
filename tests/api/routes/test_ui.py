def test_ui_index_served(test_client):
    response = test_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "DAEMON Control Panel" in response.text


def test_ui_static_assets_served(test_client):
    css_response = test_client.get("/static/css/style.css")
    js_response = test_client.get("/static/js/app.js")

    assert css_response.status_code == 200
    assert "text/css" in css_response.headers["content-type"]

    assert js_response.status_code == 200
    assert "javascript" in js_response.headers["content-type"]
