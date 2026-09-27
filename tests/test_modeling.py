import numpy as np
import pandas as pd

from surgisense.modeling import CoxModel


def test_preprocessor_is_fitted_on_training_patients_only() -> None:
    training = pd.DataFrame({
        "age": [40.0, 50.0, 60.0, np.nan, 70.0, 80.0, 55.0, 65.0],
        "sex": ["Female", "Male"] * 4,
        "stage": ["I", "II", "III", "IV"] * 2,
        "tmb": [1.0, 2.0, 3.0, 4.0] * 2,
        "duration_months": [10.0, 30.0, 15.0, 40.0, 18.0, 42.0, 13.0, 50.0],
        "event": [1, 0, 1, 0, 1, 0, 1, 0],
    })
    model = CoxModel.fit(training, include_tmb=False)
    numeric = model.preprocessor.named_transformers_["numeric"]
    assert numeric.named_steps["impute"].statistics_[0] == 60.0

    validation = training.iloc[[0]].copy()
    validation["age"] = 9999.0
    risk = model.predict_risk(validation)
    assert np.isfinite(risk).all()
    assert numeric.named_steps["impute"].statistics_[0] == 60.0


def test_clinical_model_does_not_require_tmb() -> None:
    training = pd.DataFrame({
        "age": [40.0, 50.0, 60.0, 70.0, 80.0, 55.0, 65.0, 75.0],
        "sex": ["Female", "Male"] * 4,
        "stage": ["I", "II", "III", "IV"] * 2,
        "duration_months": [10.0, 30.0, 15.0, 40.0, 18.0, 42.0, 13.0, 50.0],
        "event": [1, 0, 1, 0, 1, 0, 1, 0],
    })
    model = CoxModel.fit(training, include_tmb=False)
    assert np.isfinite(model.predict_risk(training)).all()
