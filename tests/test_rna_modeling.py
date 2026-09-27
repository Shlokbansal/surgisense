import numpy as np
import pandas as pd

from surgisense.rna_modeling import RNACoxModel, evaluate_rna


def test_gene_selection_and_scaling_only_use_training_patients() -> None:
    n = 24
    cohort = pd.DataFrame({
        "age": [40.0 + i for i in range(n)],
        "sex": ["Female", "Male"] * (n // 2),
        "stage": ["I", "II", "III", "IV"] * (n // 4),
        "duration_months": [8.0 + 2 * i for i in range(n)],
        "event": [1, 0] * (n // 2),
    })
    rows = np.arange(n, dtype=float)
    training_rna = np.column_stack([
        np.sin(rows * (i + 1) / 5) + rows * i / 20 for i in range(6)
    ])
    training_rna[:, 0] = 0.0
    model = RNACoxModel.fit(cohort, training_rna)
    assert 0 not in model.gene_indices
    means_before = model.gene_scaler.mean_.copy()
    validation_rna = training_rna[[0]].copy()
    validation_rna[0, 0] = 1_000_000.0
    risk = model.predict_risk(cohort.iloc[[0]], validation_rna)
    assert np.isfinite(risk).all()
    np.testing.assert_array_equal(model.gene_scaler.mean_, means_before)


def test_rna_evaluation_keeps_held_out_patients_separate() -> None:
    generator = np.random.default_rng(17)
    n = 80
    cohort = pd.DataFrame({
        "patient_id": [f"P{i}" for i in range(n)],
        "age": generator.integers(40, 80, size=n).astype(float),
        "sex": ["Female", "Male"] * (n // 2),
        "stage": ["I", "II", "III", "IV"] * (n // 4),
        "duration_months": generator.uniform(2, 80, size=n),
        "event": [0, 1] * (n // 2),
    })
    expression = generator.uniform(0, 12, size=(n, 8))
    report, coefficients = evaluate_rna(cohort, expression, [str(i) for i in range(8)])
    split = report["split"]
    assert set(split["development_patient_ids"]).isdisjoint(split["held_out_patient_ids"])
    assert len(split["development_patient_ids"]) == 64
    assert len(split["held_out_patient_ids"]) == 16
    assert len(report["models"]["clinical_plus_rna"]["development_cv_c_index"]) == 5
    assert len(report["development_selected_gene_ids"]) == 8
    assert "rna_pc_1" in set(coefficients["feature"])
