"""Prespecified RNA benchmark with all learned transforms fitted on training patients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from surgisense.modeling import (
    SEED,
    CoxModel,
    bootstrap_interval,
    c_index,
    make_preprocessor,
    prepare_features,
)
from surgisense.validation import HORIZONS_MONTHS, evaluate_horizons

GENE_COUNT = 1000
RNA_COMPONENTS = 5


@dataclass
class RNACoxModel:
    clinical_preprocessor: object
    gene_indices: np.ndarray
    gene_scaler: StandardScaler
    pca: PCA
    component_scaler: StandardScaler
    estimator: CoxPHFitter
    feature_names: list[str]

    @classmethod
    def fit(cls, cohort: pd.DataFrame, expression: np.ndarray) -> RNACoxModel:
        if len(cohort) != len(expression):
            raise ValueError("Clinical and RNA rows are misaligned")
        if not np.isfinite(expression).all():
            raise ValueError("RNA features must be finite")
        variance = expression.var(axis=0)
        nonconstant = np.flatnonzero(variance > 1e-8)
        if len(nonconstant) < RNA_COMPONENTS:
            raise ValueError("Too few variable genes for RNA components")
        ranked = nonconstant[np.argsort(-variance[nonconstant], kind="stable")]
        gene_indices = ranked[:GENE_COUNT]
        gene_scaler = StandardScaler().fit(expression[:, gene_indices])
        scaled_genes = gene_scaler.transform(expression[:, gene_indices])
        pca = PCA(n_components=RNA_COMPONENTS, svd_solver="randomized", random_state=SEED)
        components = pca.fit_transform(scaled_genes)
        component_scaler = StandardScaler().fit(components)
        standardized_components = component_scaler.transform(components)

        clinical_preprocessor = make_preprocessor(include_tmb=False)
        clinical = clinical_preprocessor.fit_transform(prepare_features(cohort, False))
        feature_names = clinical_preprocessor.get_feature_names_out().tolist()
        feature_names += [f"rna_pc_{i + 1}" for i in range(RNA_COMPONENTS)]
        design = pd.DataFrame(
            np.column_stack((clinical, standardized_components)), columns=feature_names
        )
        design["duration_months"] = cohort["duration_months"].to_numpy()
        design["event"] = cohort["event"].to_numpy()
        estimator = CoxPHFitter(penalizer=0.1)
        estimator.fit(design, duration_col="duration_months", event_col="event")
        return cls(
            clinical_preprocessor,
            gene_indices,
            gene_scaler,
            pca,
            component_scaler,
            estimator,
            feature_names,
        )

    def predict_risk(self, cohort: pd.DataFrame, expression: np.ndarray) -> np.ndarray:
        design = self._design(cohort, expression)
        return self.estimator.predict_partial_hazard(design).to_numpy().ravel()

    def predict_survival(
        self, cohort: pd.DataFrame, expression: np.ndarray, horizon_months: float
    ) -> np.ndarray:
        design = self._design(cohort, expression)
        survival = self.estimator.predict_survival_function(design, times=[horizon_months])
        return survival.iloc[0].to_numpy()

    def _design(self, cohort: pd.DataFrame, expression: np.ndarray) -> pd.DataFrame:
        if len(cohort) != len(expression):
            raise ValueError("Clinical and RNA rows are misaligned")
        clinical = self.clinical_preprocessor.transform(prepare_features(cohort, False))
        scaled_genes = self.gene_scaler.transform(expression[:, self.gene_indices])
        components = self.component_scaler.transform(self.pca.transform(scaled_genes))
        return pd.DataFrame(np.column_stack((clinical, components)), columns=self.feature_names)

    def proportional_hazards_p_values(
        self, cohort: pd.DataFrame, expression: np.ndarray
    ) -> dict[str, float]:
        """Exploratory Schoenfeld-residual checks on development patients only."""
        design = self._design(cohort, expression)
        design["duration_months"] = cohort["duration_months"].to_numpy()
        design["event"] = cohort["event"].to_numpy()
        diagnostic = proportional_hazard_test(
            self.estimator, design, time_transform="rank"
        ).summary
        return {feature: float(p_value) for feature, p_value in diagnostic["p"].items()}

    def coefficients(self) -> pd.DataFrame:
        coefficients = self.estimator.params_
        return pd.DataFrame({
            "feature": coefficients.index,
            "log_hazard_coefficient": coefficients.to_numpy(),
            "hazard_ratio": np.exp(coefficients.to_numpy()),
        })


def paired_delta_interval(
    cohort: pd.DataFrame, clinical_risk: np.ndarray, rna_risk: np.ndarray,
    repetitions: int = 500,
) -> list[float]:
    """Resample the same held-out patients for both model C-indices."""
    generator = np.random.default_rng(SEED)
    differences = []
    for _ in range(repetitions):
        rows = generator.integers(0, len(cohort), len(cohort))
        try:
            differences.append(
                c_index(cohort.iloc[rows], rna_risk[rows])
                - c_index(cohort.iloc[rows], clinical_risk[rows])
            )
        except ZeroDivisionError:
            continue
    if len(differences) < repetitions // 2:
        raise ValueError("Too few comparable bootstrap samples for a paired interval")
    return np.quantile(differences, [0.025, 0.975]).tolist()


def evaluate_rna(
    cohort: pd.DataFrame, expression: np.ndarray, gene_ids: list[str]
) -> tuple[dict, pd.DataFrame]:
    """Compare clinical-only and RNA-augmented Cox models on identical patients."""
    if len(cohort) != len(expression) or expression.shape[1] != len(gene_ids):
        raise ValueError("Clinical patients, RNA rows, and gene IDs must align")
    train_idx, test_idx = train_test_split(
        np.arange(len(cohort)), test_size=0.2, random_state=SEED, stratify=cohort["event"]
    )
    development = cohort.iloc[train_idx].reset_index(drop=True)
    test = cohort.iloc[test_idx].reset_index(drop=True)
    development_rna = expression[train_idx]
    test_rna = expression[test_idx]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clinical_cv = []
    rna_cv = []
    for fit_idx, validation_idx in folds.split(development, development["event"]):
        fit = development.iloc[fit_idx]
        validation = development.iloc[validation_idx]
        clinical = CoxModel.fit(fit, include_tmb=False)
        rna = RNACoxModel.fit(fit, development_rna[fit_idx])
        clinical_cv.append(c_index(validation, clinical.predict_risk(validation)))
        rna_cv.append(
            c_index(
                validation,
                rna.predict_risk(validation, development_rna[validation_idx]),
            )
        )
    clinical = CoxModel.fit(development, include_tmb=False)
    rna = RNACoxModel.fit(development, development_rna)
    clinical_risk = clinical.predict_risk(test)
    rna_risk = rna.predict_risk(test, test_rna)
    clinical_test = c_index(test, clinical_risk)
    rna_test = c_index(test, rna_risk)
    report = {
        "random_seed": SEED,
        "development_patients": len(development),
        "held_out_patients": len(test),
        "development_deaths": int(development["event"].sum()),
        "held_out_deaths": int(test["event"].sum()),
        "split": {
            "development_patient_ids": development["patient_id"].tolist(),
            "held_out_patient_ids": test["patient_id"].tolist(),
        },
        "model_specification": {
            "clinical_features": ["age", "sex", "stage"],
            "rna_transform": "log2(RSEM + 1)",
            "gene_selection": f"top {GENE_COUNT} by training-only variance",
            "gene_scaling": "training-only standardization",
            "dimension_reduction": f"{RNA_COMPONENTS} training-only principal components",
            "component_scaling": "training-only standardization",
            "estimator": "lifelines.CoxPHFitter",
            "penalizer": 0.1,
            "cross_validation_folds": 5,
            "held_out_fraction": 0.2,
        },
        "metric": "Harrell concordance index; larger is better, 0.5 is chance",
        "models": {
            "clinical": {
                "development_cv_c_index": clinical_cv,
                "development_cv_mean_c_index": float(np.mean(clinical_cv)),
                "held_out_c_index": clinical_test,
                "held_out_bootstrap_95pct_interval": bootstrap_interval(test, clinical_risk),
            },
            "clinical_plus_rna": {
                "development_cv_c_index": rna_cv,
                "development_cv_mean_c_index": float(np.mean(rna_cv)),
                "held_out_c_index": rna_test,
                "held_out_bootstrap_95pct_interval": bootstrap_interval(test, rna_risk),
                "development_proportional_hazards_p_values": (
                    rna.proportional_hazards_p_values(development, development_rna)
                ),
            },
        },
        "held_out_rna_minus_clinical_c_index": rna_test - clinical_test,
        "held_out_paired_delta_bootstrap_95pct_interval": paired_delta_interval(
            test, clinical_risk, rna_risk
        ),
        "development_selected_gene_ids": [gene_ids[i] for i in rna.gene_indices],
        "development_rna_pc_explained_variance_ratio": rna.pca.explained_variance_ratio_.tolist(),
        "fixed_horizon_validation": evaluate_horizons(
            development,
            test,
            {
                "clinical": {
                    horizon: clinical.predict_survival(test, horizon) for horizon in HORIZONS_MONTHS
                },
                "clinical_plus_rna": {
                    horizon: rna.predict_survival(test, test_rna, horizon)
                    for horizon in HORIZONS_MONTHS
                },
            },
        ),
    }
    return report, rna.coefficients()
