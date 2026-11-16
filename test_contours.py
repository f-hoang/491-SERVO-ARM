"""
Tests for contours.py. No camera needed: everything uses synthetic images.
Run:  pip install pytest opencv-python numpy
      pytest -v test_contours.py
"""
import cv2
import numpy as np
import pytest

from contours import MODES, preprocess, classify, draw_objects

W, H = 640, 480


def blank():
    """White background, like a table top."""
    return np.full((H, W, 3), 255, dtype=np.uint8)


def with_shape(kind):
    """Return (frame, contour) with one dark shape on a white background."""
    f = blank()
    if kind == "square":
        cv2.rectangle(f, (200, 150), (320, 270), (0, 0, 0), -1)
    elif kind == "rectangle":
        cv2.rectangle(f, (200, 150), (400, 250), (0, 0, 0), -1)
    elif kind == "triangle":
        pts = np.array([[300, 120], [200, 300], [400, 300]])
        cv2.fillPoly(f, [pts], (0, 0, 0))
    elif kind == "circle":
        cv2.circle(f, (300, 220), 70, (0, 0, 0), -1)
    mask = preprocess(f, "otsu")
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    return f, mask, c


# ---------- preprocess ----------

@pytest.mark.parametrize("mode", MODES)
def test_mask_shape_and_dtype(mode):
    f, _, _ = with_shape("square")
    mask = preprocess(f, mode)
    assert mask.shape == (H, W)
    assert mask.dtype == np.uint8


@pytest.mark.parametrize("mode", MODES)
def test_mask_is_binary(mode):
    f, _, _ = with_shape("circle")
    mask = preprocess(f, mode)
    assert set(np.unique(mask)).issubset({0, 255})


def test_otsu_dark_object_becomes_white():
    f, mask, _ = with_shape("square")
    assert mask[210, 260] == 255   # inside the object
    assert mask[10, 10] == 0       # background


@pytest.mark.parametrize("mode", MODES)
def test_blank_frame_has_no_blobs(mode):
    mask = preprocess(blank(), mode)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in cnts if cv2.contourArea(c) > 800]
    assert big == []


def test_stray_pixel_filtered_by_min_area():
    # Otsu can blow a single dark pixel up into a small blob on a uniform frame,
    # so the real protection is the min_area filter in draw_objects.
    f = blank()
    f[100, 100] = (0, 0, 0)
    mask = preprocess(f, "otsu")
    assert draw_objects(f.copy(), mask, 800) == 0


# ---------- classify ----------

@pytest.mark.parametrize("kind,expected", [
    ("square", "square"),
    ("rectangle", "rectangle"),
    ("triangle", "triangle"),
    ("circle", "circle/ellipse"),
])
def test_classify_shapes(kind, expected):
    _, _, c = with_shape(kind)
    name, approx = classify(c)
    assert name == expected
    assert len(approx) >= 3


# ---------- draw_objects ----------

def test_counts_one_object():
    f, mask, _ = with_shape("square")
    assert draw_objects(f.copy(), mask, 800) == 1


def test_counts_multiple_objects():
    f = blank()
    cv2.rectangle(f, (50, 50), (150, 150), (0, 0, 0), -1)
    cv2.circle(f, (400, 300), 60, (0, 0, 0), -1)
    mask = preprocess(f, "otsu")
    assert draw_objects(f.copy(), mask, 800) == 2


def test_min_area_filters_small_objects():
    f = blank()
    cv2.rectangle(f, (50, 50), (80, 80), (0, 0, 0), -1)  # ~900 px
    mask = preprocess(f, "otsu")
    assert draw_objects(f.copy(), mask, 100) >= 1
    assert draw_objects(f.copy(), mask, 50000) == 0


def test_draw_modifies_frame():
    f, mask, _ = with_shape("square")
    view = f.copy()
    draw_objects(view, mask, 800)
    assert not np.array_equal(view, f)


def test_empty_mask_returns_zero():
    f = blank()
    mask = np.zeros((H, W), dtype=np.uint8)
    assert draw_objects(f, mask, 800) == 0


def test_centroid_near_true_center():
    f, mask, c = with_shape("square")
    m = cv2.moments(c)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    assert abs(cx - 260) < 3 and abs(cy - 210) < 3
