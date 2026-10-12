"""Test the five application behaviors required by Assignment 2."""

from io import BytesIO
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest

import app as app_module
import model as model_module


@pytest.fixture
def web_client(monkeypatch):
    """Provide a Flask test client for each test."""
    monkeypatch.setitem(app_module.app.config, "TESTING", True)
    with app_module.app.test_client() as client:
        yield client


def make_image():
    """Create a valid RGB PNG image in memory."""
    stream = BytesIO()
    Image.new("RGB", (16, 16), "white").save(stream, format="PNG")
    stream.seek(0)
    return stream


def test_home_page(web_client):
    """The home page responds successfully and contains the upload input."""
    response = web_client.get("/")
    assert response.status_code == 200
    assert b'name="file"' in response.data


def test_valid_upload(web_client, monkeypatch):
    """A valid image produces the expected mocked prediction."""
    prediction = Mock(return_value=9)
    monkeypatch.setattr(app_module, "predict_result", prediction)

    response = web_client.post(
        "/prediction",
        data={"file": (make_image(), "sign.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert ">9</h2>" in response.get_data(as_text=True)
    prediction.assert_called_once()


def test_missing_file(web_client, monkeypatch):
    """A missing upload displays an error without invoking prediction."""
    prediction = Mock()
    monkeypatch.setattr(app_module, "predict_result", prediction)

    response = web_client.post("/prediction", data={})

    assert response.status_code == 200
    assert b"File cannot be processed." in response.data
    prediction.assert_not_called()


def test_invalid_image(web_client, monkeypatch):
    """Invalid image bytes display an error without invoking prediction."""
    prediction = Mock()
    monkeypatch.setattr(app_module, "predict_result", prediction)

    response = web_client.post(
        "/prediction",
        data={"file": (BytesIO(b"not an image"), "invalid.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"File cannot be processed." in response.data
    prediction.assert_not_called()


def test_preprocessing():
    """Preprocessing resizes an RGB image and normalizes its pixels."""
    batch = model_module.preprocess_img(make_image())
    assert batch.shape == (1, 224, 224, 3)
    np.testing.assert_allclose(batch, 1.0)
