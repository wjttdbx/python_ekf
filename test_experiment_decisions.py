"""Regression tests for experiment decision rules."""

import csv

from experiments.e2_gono_go import _gono_go_decision


def _write_decision_rows(path, cases):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["geometry", "sigma_theta_deg", "T_FP", "sustained"],
        )
        writer.writeheader()
        for geometry, low_tfp, high_tfp, low_sustain, high_sustain in cases:
            writer.writerow({
                "geometry": geometry,
                "sigma_theta_deg": 0.001,
                "T_FP": low_tfp,
                "sustained": low_sustain,
            })
            writer.writerow({
                "geometry": geometry,
                "sigma_theta_deg": 0.1,
                "T_FP": high_tfp,
                "sustained": high_sustain,
            })


def test_gono_go_requires_speed_and_softness_in_same_geometries(tmp_path):
    csv_path = tmp_path / "mismatched_evidence.csv"
    _write_decision_rows(csv_path, [
        ("radial", 100.0, 80.0, False, False),
        ("alongtrack", 100.0, 80.0, True, False),
        ("crosstrack", 80.0, 100.0, True, False),
    ])

    assert _gono_go_decision(csv_path) is False


def test_gono_go_accepts_two_same_geometry_inversions(tmp_path):
    csv_path = tmp_path / "coincident_evidence.csv"
    _write_decision_rows(csv_path, [
        ("radial", 100.0, 80.0, True, False),
        ("alongtrack", 100.0, 80.0, True, False),
        ("crosstrack", 80.0, 100.0, True, True),
    ])

    assert _gono_go_decision(csv_path) is True