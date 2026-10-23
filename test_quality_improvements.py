
"""Additional regression tests for image recognition quality."""

from io import BytesIO
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from model import preprocess_img, predict_result


def create_test_image(color="white", image_format="PNG"):
    """Create a real image in memory for upload testing."""
    image = Image.new("RGB", (64, 64), color=color)
    image_bytes = BytesIO()
    image.save(image_bytes, format=image_format)
    image_bytes.seek(0)
    return image_bytes


def test_preprocessing_output_shape():
    """Verify that image preprocessing produces the expected shape."""
    image = create_test_image()
    processed = preprocess_img(image)

    assert processed.shape == (1, 224, 224, 3)


def test_preprocessing_normalization():
    """Verify that pixel values are normalized to the expected range."""
    image = create_test_image(color="white")
    processed = preprocess_img(image)

    assert np.min(processed) >= 0
    assert np.max(processed) <= 1
    assert np.allclose(processed, 1.0)


def test_preprocessing_invalid_image():
    """Verify that invalid image data raises an appropriate exception."""
    invalid_image = BytesIO(b"not_a_real_image")

    with pytest.raises(OSError):
        preprocess_img(invalid_image)


def test_prediction_selects_highest_score():
    """Verify that the predicted class has the highest confidence."""
    fake_scores = np.array([[0.1, 0.2, 0.05, 0.65]])

    with patch("model.model.predict", return_value=fake_scores):
        prediction = predict_result(np.zeros((1, 224, 224, 3)))

    assert prediction == 3


def test_valid_image_upload(client):
    """Verify that a valid image reaches the prediction result page."""
    image = create_test_image()

    with patch("app.predict_result", return_value=6):
        response = client.post(
            "/prediction",
            data={"file": (image, "test.png")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    assert b"File cannot be processed" not in response.data
    assert b"6" in response.data


def test_invalid_image_upload(client):
    """Verify that invalid uploaded image data displays an error."""
    response = client.post(
        "/prediction",
        data={"file": (BytesIO(b"invalid"), "invalid.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"File cannot be processed" in response.data


def test_missing_image_upload(client):
    """Verify that a missing uploaded file displays an error."""
    response = client.post(
        "/prediction",
        data={},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"File cannot be processed" in response.data
