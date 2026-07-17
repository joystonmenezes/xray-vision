"""End-to-end tests: preprocessing, model inference, and the API."""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.preprocessing import prepare_input

REPO = Path(__file__).resolve().parent.parent
EXAMPLE = REPO / "docs" / "example_input.png"


def test_prepare_input_shape_and_range():
    img = Image.open(EXAMPLE)
    batch = prepare_input(img)
    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    assert 0.0 <= batch.min() and batch.max() <= 1.0


def test_prepare_input_handles_grayscale_and_odd_sizes():
    img = Image.new("L", (301, 97), color=128)
    batch = prepare_input(img)
    assert batch.shape == (1, 224, 224, 3)


@pytest.fixture(scope="session")
def model():
    from app.inference import XRayModel

    return XRayModel()


def test_predict_probabilities_sum_to_one(model):
    batch = prepare_input(Image.open(EXAMPLE))
    prob = model.predict(batch)[0]
    assert prob.shape == (2,)
    assert abs(prob.sum() - 1.0) < 1e-4


def test_example_image_is_abnormal(model):
    """The bundled example is a known abnormal study (~99% in the 2021 app)."""
    batch = prepare_input(Image.open(EXAMPLE))
    prob = model.predict(batch)[0]
    assert prob[1] > 0.9


def test_tta_returns_valid_distribution(model):
    batch = prepare_input(Image.open(EXAMPLE))
    prob = model.predict_tta(batch)
    assert prob.shape == (2,)
    assert abs(prob.sum() - 1.0) < 1e-4
    assert prob[1] > 0.9  # example stays abnormal under multi-view averaging


def test_tta_views_are_all_valid_images(model):
    """Every generated view is a well-formed [0,1] 224x224x3 image."""
    from app.inference import _tta_views

    base = prepare_input(Image.open(EXAMPLE))[0]
    views = _tta_views(base)
    assert views.shape[1:] == (224, 224, 3)
    assert views.shape[0] >= 4  # original + flip + rotations + zoom
    assert 0.0 <= views.min() and views.max() <= 1.0
    # View 0 must be the untouched original (callers rely on this).
    assert np.array_equal(views[0], base)


def test_tta_agrees_with_single_view(model):
    """TTA refines, it should not contradict, the single-view prediction."""
    batch = prepare_input(Image.open(EXAMPLE))
    single = model.predict(batch)[0][1]
    tta = model.predict_tta(batch)[1]
    assert abs(single - tta) < 0.1  # same ballpark, same verdict


def test_smoothgrad_produces_localized_map(model):
    batch = prepare_input(Image.open(EXAMPLE))
    mask = model.smoothgrad(batch)
    assert mask.shape == (224, 224)
    assert mask.max() > 0
    # 99th-percentile thresholding keeps the map sparse.
    assert (mask > 0).mean() < 0.10


def test_api_analyze_endpoint(model):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        buf = io.BytesIO()
        Image.open(EXAMPLE).convert("RGB").save(buf, format="PNG")
        res = client.post(
            "/api/analyze",
            files={"file": ("xray.png", buf.getvalue(), "image/png")},
        )
        assert res.status_code == 200
        data = res.json()
        assert 0 <= data["abnormal_probability"] <= 100
        assert data["input_image"].startswith("data:image/png;base64,")
        assert data["heatmap"].startswith("data:image/png;base64,")


def test_api_rejects_non_image(model):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        res = client.post(
            "/api/analyze",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
        assert res.status_code == 415
