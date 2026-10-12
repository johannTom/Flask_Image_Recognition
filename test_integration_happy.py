"""Integration tests for the image prediction route."""

from io import BytesIO
import re


def test_successful_prediction(client):
    """A successful upload returns a numeric prediction."""
    img_data = BytesIO(b"fake_image_data")

    response = client.post(
        "/prediction",
        data={"file": (img_data, "test.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert re.search(rb"<h2[^>]*>\s*[0-9]\s*</h2>", response.data)
