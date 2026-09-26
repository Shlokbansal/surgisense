from pathlib import Path

import pandas as pd
import pytest

from surgisense import cohort as cohort_module


def test_cohort_filters_and_censoring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patients = pd.DataFrame({
        "PATIENT_ID": ["p1", "p2", "p3", "p4"],
        "CANCER_TYPE_ACRONYM": ["LUAD"] * 4,
        "AGE": ["60", "70", "80", "90"],
        "SEX": ["Female", "Male", "Male", "Female"],
        "AJCC_PATHOLOGIC_TUMOR_STAGE": ["STAGE IA", "STAGE IIIB", "STAGE IV", "STAGE I"],
        "OS_STATUS": ["1:DECEASED", "0:LIVING", "0:LIVING", "1:DECEASED"],
        "OS_MONTHS": ["12", "25", "0", "bad"],
    })
    samples = pd.DataFrame({
        "PATIENT_ID": ["p1", "p2", "p3", "p4"],
        "SAMPLE_ID": ["s1", "s2", "s3", "s4"],
        "ONCOTREE_CODE": ["LUAD"] * 4,
        "SAMPLE_TYPE": ["Primary"] * 4,
        "TMB_NONSYNONYMOUS": ["3", "4", "5", "6"],
    })
    patient_path, sample_path = tmp_path / "patient.txt", tmp_path / "sample.txt"
    patients.to_csv(patient_path, sep="\t", index=False)
    samples.to_csv(sample_path, sep="\t", index=False)
    monkeypatch.setattr(
        cohort_module,
        "verify_sources",
        lambda _: {
            "data_clinical_patient.txt": patient_path,
            "data_clinical_sample.txt": sample_path,
        },
    )

    result, audit = cohort_module.build_cohort(tmp_path)
    assert result["patient_id"].tolist() == ["p1", "p2"]
    assert result["event"].tolist() == [1, 0]
    assert result["stage"].tolist() == ["I", "III"]
    assert audit["excluded_missing_or_invalid_survival"] == 2


def test_rejects_multiple_samples_per_patient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patients = pd.DataFrame({
        "PATIENT_ID": ["p1"], "CANCER_TYPE_ACRONYM": ["LUAD"], "AGE": ["60"],
        "SEX": ["Female"], "AJCC_PATHOLOGIC_TUMOR_STAGE": ["STAGE I"],
        "OS_STATUS": ["0:LIVING"], "OS_MONTHS": ["12"],
    })
    samples = pd.DataFrame({
        "PATIENT_ID": ["p1", "p1"], "SAMPLE_ID": ["s1", "s2"],
        "ONCOTREE_CODE": ["LUAD", "LUAD"], "SAMPLE_TYPE": ["Primary", "Primary"],
        "TMB_NONSYNONYMOUS": ["3", "4"],
    })
    patient_path, sample_path = tmp_path / "patient.txt", tmp_path / "sample.txt"
    patients.to_csv(patient_path, sep="\t", index=False)
    samples.to_csv(sample_path, sep="\t", index=False)
    monkeypatch.setattr(
        cohort_module,
        "verify_sources",
        lambda _: {
            "data_clinical_patient.txt": patient_path,
            "data_clinical_sample.txt": sample_path,
        },
    )
    with pytest.raises(ValueError, match="Multiple samples"):
        cohort_module.build_cohort(tmp_path)
