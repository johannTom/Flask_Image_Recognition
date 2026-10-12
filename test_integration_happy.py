"""Integration tests for the image prediction route."""

from io import BytesIO
import re

from PIL import Image


def test_successful_prediction(client):
    """A successful upload returns a numeric prediction."""
    img_data = BytesIO()
    Image.new("RGB", (16, 16), "white").save(img_data, format="PNG")
    img_data.seek(0)

    response = client.post(
        "/prediction",
        data={"file": (img_data, "test.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert re.search(rb"<h2[^>]*>\s*[0-9]\s*</h2>", response.data)
