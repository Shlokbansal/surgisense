import numpy as np
import pandas as pd
import pytest

from surgisense.validation import evaluate_horizons, ipcw_brier_score


def test_ipcw_brier_matches_mean_squared_error_without_censoring() -> None:
    development = pd.DataFrame({
        "duration_months": [2.0, 4.0, 6.0], "event": [1, 1, 1]
    })
    test = pd.DataFrame({
        "duration_months": [2.0, 6.0], "event": [1, 1]
    })
    prediction = np.array([0.2, 0.7])
    expected = (0.2**2 + (1 - 0.7) ** 2) / 2
    assert ipcw_brier_score(development, test, prediction, 4.0) == pytest.approx(expected)


def test_ipcw_brier_excludes_unknown_outcomes_and_weights_observed() -> None:
    development = pd.DataFrame({
        "duration_months": [2.0, 4.0, 6.0, 8.0], "event": [1, 0, 1, 0]
    })
    test = pd.DataFrame({
        "duration_months": [2.0, 3.0, 4.0, 6.0], "event": [1, 0, 1, 0]
    })
    prediction = np.array([0.1, 0.2, 0.3, 0.4])
    expected = (0.1**2 + 0 + 0.3**2 / (2 / 3) + (1 - 0.4) ** 2 / (2 / 3)) / 4
    assert ipcw_brier_score(development, test, prediction, 5.0) == pytest.approx(expected)


def test_ipcw_brier_rejects_invalid_predictions() -> None:
    cohort = pd.DataFrame({"duration_months": [1.0, 2.0], "event": [1, 0]})
    with pytest.raises(ValueError, match="between zero and one"):
        ipcw_brier_score(cohort, cohort, np.array([-0.1, 0.7]), 1.0)
    with pytest.raises(ValueError, match="one finite value"):
        ipcw_brier_score(cohort, cohort, np.array([0.5]), 1.0)


def test_horizon_report_includes_constant_baseline() -> None:
    development = pd.DataFrame({
        "duration_months": [3.0, 15.0, 30.0, 40.0], "event": [1, 1, 0, 1]
    })
    test = pd.DataFrame({
        "duration_months": [4.0, 18.0, 31.0, 42.0], "event": [1, 1, 0, 1]
    })
    prediction = {"model": {12.0: np.full(4, 0.8), 24.0: np.full(4, 0.7)}}
    report = evaluate_horizons(development, test, prediction)
    assert set(report) == {"12", "24"}
    assert report["12"]["held_out_deaths_by_horizon"] == 1
    assert 0 <= report["24"]["development_km_constant_ipcw_brier_score"] <= 1
    assert report["24"]["models"]["model"]["mean_predicted_survival"] == pytest.approx(0.7)
