"""Prespecified Cox survival benchmarks with fold-local preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
NUMERIC_CLINICAL = ["age"]
NUMERIC_MULTIMODAL = ["age", "log1p_tmb"]
CATEGORICAL = ["sex", "stage"]


def prepare_features(cohort: pd.DataFrame, include_tmb: bool) -> pd.DataFrame:
    features = cohort[["age", "sex", "stage"]].copy()
    if include_tmb:
        features["log1p_tmb"] = np.log1p(cohort["tmb"].astype(float))
    for column in CATEGORICAL:
        features[column] = features[column].astype(object).where(features[column].notna(), np.nan)
    return features


def make_preprocessor(include_tmb: bool) -> ColumnTransformer:
    numeric = NUMERIC_MULTIMODAL if include_tmb else NUMERIC_CLINICAL
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                numeric,
            ),
            (
                "category",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(
                            categories=[["Female", "Male"], ["I", "II", "III", "IV"]],
                            drop="first",
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]),
                CATEGORICAL,
            ),
        ],
        verbose_feature_names_out=False,
    )


@dataclass
class CoxModel:
    preprocessor: ColumnTransformer
    estimator: CoxPHFitter
    feature_names: list[str]
    include_tmb: bool

    @classmethod
    def fit(cls, cohort: pd.DataFrame, include_tmb: bool) -> CoxModel:
        preprocessor = make_preprocessor(include_tmb)
        array = preprocessor.fit_transform(prepare_features(cohort, include_tmb))
        names = preprocessor.get_feature_names_out().tolist()
        transformed = pd.DataFrame(array, columns=names, index=cohort.index)
        transformed["duration_months"] = cohort["duration_months"].to_numpy()
        transformed["event"] = cohort["event"].to_numpy()
        estimator = CoxPHFitter(penalizer=0.1)
        estimator.fit(transformed, duration_col="duration_months", event_col="event")
        return cls(preprocessor, estimator, names, include_tmb)

    def predict_risk(self, cohort: pd.DataFrame) -> np.ndarray:
        array = self.preprocessor.transform(prepare_features(cohort, self.include_tmb))
        transformed = pd.DataFrame(array, columns=self.feature_names, index=cohort.index)
        return self.estimator.predict_partial_hazard(transformed).to_numpy().ravel()

    def coefficients(self) -> pd.DataFrame:
        coefficients = self.estimator.params_
        return pd.DataFrame({
            "feature": coefficients.index,
            "log_hazard_coefficient": coefficients.to_numpy(),
            "hazard_ratio": np.exp(coefficients.to_numpy()),
        })

    def proportional_hazards_p_values(self, cohort: pd.DataFrame) -> dict[str, float]:
        """Exploratory Schoenfeld-residual checks on the development cohort only."""
        array = self.preprocessor.transform(prepare_features(cohort, self.include_tmb))
        transformed = pd.DataFrame(array, columns=self.feature_names, index=cohort.index)
        transformed["duration_months"] = cohort["duration_months"].to_numpy()
        transformed["event"] = cohort["event"].to_numpy()
        diagnostic = proportional_hazard_test(
            self.estimator, transformed, time_transform="rank"
        ).summary
        return {feature: float(p_value) for feature, p_value in diagnostic["p"].items()}


def c_index(cohort: pd.DataFrame, risk: np.ndarray) -> float:
    return float(concordance_index(cohort["duration_months"], -risk, cohort["event"]))


def bootstrap_interval(
    cohort: pd.DataFrame, risk: np.ndarray, repetitions: int = 500
) -> list[float]:
    """Percentile interval for held-out discrimination, conditional on the fitted model."""
    generator = np.random.default_rng(SEED)
    scores = []
    for _ in range(repetitions):
        rows = generator.integers(0, len(cohort), len(cohort))
        try:
            scores.append(c_index(cohort.iloc[rows], risk[rows]))
        except ZeroDivisionError:
            continue  # A bootstrap sample can have no comparable pairs.
    if len(scores) < repetitions // 2:
        raise ValueError("Too few comparable bootstrap samples for an interval")
    return np.quantile(scores, [0.025, 0.975]).tolist()


def evaluate(cohort: pd.DataFrame) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Cross-validate on development data, then evaluate both fixed models once on test data."""
    train_idx, test_idx = train_test_split(
        np.arange(len(cohort)), test_size=0.2, random_state=SEED, stratify=cohort["event"]
    )
    development = cohort.iloc[train_idx].reset_index(drop=True)
    test = cohort.iloc[test_idx].reset_index(drop=True)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    report = {
        "random_seed": SEED,
        "split": {
            "development_patient_ids": development["patient_id"].tolist(),
            "held_out_patient_ids": test["patient_id"].tolist(),
        },
        "development_patients": len(development),
        "held_out_patients": len(test),
        "development_deaths": int(development["event"].sum()),
        "held_out_deaths": int(test["event"].sum()),
        "metric": "Harrell concordance index; larger is better, 0.5 is chance",
        "model_specification": {
            "estimator": "lifelines.CoxPHFitter",
            "penalizer": 0.1,
            "clinical_features": ["age", "sex", "stage"],
            "additional_feature": "log1p(TMB)",
            "cross_validation_folds": 5,
            "held_out_fraction": 0.2,
        },
        "models": {},
    }
    coefficients = {}
    for name, include_tmb in (("clinical", False), ("clinical_plus_tmb", True)):
        fold_scores = []
        for fit_idx, validation_idx in folds.split(development, development["event"]):
            fit = development.iloc[fit_idx]
            validation = development.iloc[validation_idx]
            model = CoxModel.fit(fit, include_tmb)
            fold_scores.append(c_index(validation, model.predict_risk(validation)))
        final_model = CoxModel.fit(development, include_tmb)
        test_risk = final_model.predict_risk(test)
        report["models"][name] = {
            "development_cv_c_index": fold_scores,
            "development_cv_mean_c_index": float(np.mean(fold_scores)),
            "held_out_c_index": c_index(test, test_risk),
            "held_out_bootstrap_95pct_interval": bootstrap_interval(test, test_risk),
            "development_proportional_hazards_p_values": (
                final_model.proportional_hazards_p_values(development)
            ),
        }
        coefficients[name] = final_model.coefficients()
    return report, coefficients
