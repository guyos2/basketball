import numpy as np
import pytest

from courtvision import court
from courtvision.calibration import Calibration


@pytest.mark.parametrize("xy, expected", [
    ((7.5, 1.8), "restricted"),
    ((6.0, 4.0), "paint"),
    ((10.4, 5.4), "midrange"),
    ((0.5, 1.0), "corner_three"),
    ((14.5, 2.5), "corner_three"),
    ((1.2, 2.0), "midrange"),  # inside the corner line
    ((7.5, 8.6), "above_break_three"),
    ((7.5, 10.5), "deep_three"),
])
def test_zone(xy, expected):
    assert court.zone(xy) == expected


def test_corner_line_meets_arc():
    assert court.distance_to_basket((court.CORNER_THREE_X, court.CORNER_THREE_Y)) == pytest.approx(court.THREE_RADIUS)


def test_calibration_round_trip(tmp_path):
    calib = Calibration(points={
        "baseline_left_corner": (330, 170), "baseline_right_corner": (950, 170),
        "midcourt_left": (40, 700), "midcourt_right": (1240, 700),
    }, rim_px=(640, 120))
    assert calib.reprojection_error < 1e-6
    court_pts = np.array([[7.5, 1.575], [3.0, 9.0], [14.0, 0.5]])
    np.testing.assert_allclose(calib.to_court(calib.to_image(court_pts)), court_pts, atol=1e-6)
    # The baseline is farther from the camera, so one meter there spans fewer pixels.
    far = calib.to_image([[7.0, 0.0], [8.0, 0.0]])
    near = calib.to_image([[7.0, 14.0], [8.0, 14.0]])
    assert np.linalg.norm(far[1] - far[0]) < np.linalg.norm(near[1] - near[0])

    calib.save(tmp_path / "c.json")
    loaded = Calibration.load(tmp_path / "c.json")
    assert loaded.rim_px == (640, 120)
    np.testing.assert_allclose(loaded.H, calib.H)


def test_calibration_rejects_bad_input():
    with pytest.raises(ValueError, match="Unknown landmarks"):
        Calibration(points={"free_throw_spot": (0, 0), "midcourt_left": (1, 1), "midcourt_right": (2, 2),
                            "baseline_left_corner": (3, 3)})
    with pytest.raises(ValueError, match="At least 4"):
        Calibration(points={"midcourt_left": (1, 1), "midcourt_right": (2, 2), "baseline_left_corner": (3, 3)})
