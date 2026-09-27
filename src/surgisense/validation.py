"""Fixed-horizon survival validation that accounts for right-censored follow-up."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

HORIZONS_MONTHS = (12.0, 24.0)


def _fit_censoring_curve(development: pd.DataFrame) -> KaplanMeierFitter:
    """Estimate the chance of still being observed, using development patients only."""
    return KaplanMeierFitter().fit(
        development["duration_months"], event_observed=1 - development["event"]
    )


def ipcw_brier_score(
    development: pd.DataFrame,
    test: pd.DataFrame,
    predicted_survival: np.ndarray,
    horizon_months: float,
) -> float:
    """Brier score with inverse-probability-of-censoring weights.

    Observed deaths by the horizon contribute squared predicted survival;
    patients known alive past it contribute squared predicted death probability.
    Earlier-censored patients contribute zero because their status is unknown.
    """
    prediction = np.asarray(predicted_survival, dtype=float)
    if prediction.shape != (len(test),) or not np.isfinite(prediction).all():
        raise ValueError("Predicted survival must be one finite value per test patient")
    if ((prediction < 0) | (prediction > 1)).any():
        raise ValueError("Predicted survival probabilities must lie between zero and one")
    if horizon_months <= 0:
        raise ValueError("Validation horizon must be positive")

    duration = test["duration_months"].to_numpy(dtype=float)
    event = test["event"].to_numpy(dtype=int)
    died = (duration <= horizon_months) & (event == 1)
    known_alive = duration > horizon_months
    censor_curve = _fit_censoring_curve(development)
    at_horizon = float(censor_curve.survival_function_at_times(horizon_months).iloc[0])
    if at_horizon <= 0:
        raise ValueError("Censoring survival reached zero by the validation horizon")
    contributions = np.zeros(len(test), dtype=float)
    if died.any():
        at_death = censor_curve.survival_function_at_times(duration[died]).to_numpy()
        if (at_death <= 0).any():
            raise ValueError("Censoring survival reached zero at an observed death")
        contributions[died] = prediction[died] ** 2 / at_death
    contributions[known_alive] = (1 - prediction[known_alive]) ** 2 / at_horizon
    return float(contributions.mean())


def evaluate_horizons(
    development: pd.DataFrame,
    test: pd.DataFrame,
    predictions: dict[str, dict[float, np.ndarray]],
) -> dict[str, dict]:
    """Compare model probabilities against a development-only Kaplan-Meier baseline."""
    development_km = KaplanMeierFitter().fit(
        development["duration_months"], event_observed=development["event"]
    )
    test_km = KaplanMeierFitter().fit(test["duration_months"], event_observed=test["event"])
    report = {}
    for horizon in HORIZONS_MONTHS:
        baseline = float(development_km.survival_function_at_times(horizon).iloc[0])
        observed = float(test_km.survival_function_at_times(horizon).iloc[0])
        by_model = {}
        for name, at_times in predictions.items():
            predicted = np.asarray(at_times[horizon], dtype=float)
            by_model[name] = {
                "ipcw_brier_score": ipcw_brier_score(development, test, predicted, horizon),
                "mean_predicted_survival": float(predicted.mean()),
                "mean_predicted_death": float(1 - predicted.mean()),
            }
        report[str(int(horizon))] = {
            "horizon_months": int(horizon),
            "held_out_observed_km_survival": observed,
            "held_out_observed_km_death": 1 - observed,
            "development_km_constant_survival": baseline,
            "development_km_constant_ipcw_brier_score": ipcw_brier_score(
                development, test, np.full(len(test), baseline), horizon
            ),
            "held_out_deaths_by_horizon": int(
                ((test["duration_months"] <= horizon) & (test["event"] == 1)).sum()
            ),
            "held_out_censored_by_horizon": int(
                ((test["duration_months"] <= horizon) & (test["event"] == 0)).sum()
            ),
            "models": by_model,
        }
    return report
