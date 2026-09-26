"""Validate and assemble a patient-level TCGA LUAD survival cohort."""

from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from surgisense.sources import verify_sources

PATIENT_COLUMNS = {
    "PATIENT_ID", "CANCER_TYPE_ACRONYM", "AGE", "SEX",
    "AJCC_PATHOLOGIC_TUMOR_STAGE", "OS_STATUS", "OS_MONTHS",
}
SAMPLE_COLUMNS = {
    "PATIENT_ID", "SAMPLE_ID", "ONCOTREE_CODE", "SAMPLE_TYPE", "TMB_NONSYNONYMOUS",
}
STAGE_PATTERN = re.compile(r"^STAGE\s+(IV|III|II|I)(?:[ABC])?$", re.IGNORECASE)


def _read_table(path: Path, required: set[str], unique_key: str) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", comment="#", dtype="string")
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path.name} lacks columns: {sorted(missing)}")
    if table[unique_key].isna().any() or table[unique_key].duplicated().any():
        raise ValueError(f"{path.name} has null or duplicate {unique_key} values")
    return table


def _stage(value: object) -> str | None:
    if pd.isna(value):
        return None
    match = STAGE_PATTERN.fullmatch(str(value).strip())
    return match.group(1).upper() if match else None


def build_cohort(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    paths = verify_sources(raw_dir)
    patients = _read_table(paths["data_clinical_patient.txt"], PATIENT_COLUMNS, "PATIENT_ID")
    samples = _read_table(paths["data_clinical_sample.txt"], SAMPLE_COLUMNS, "SAMPLE_ID")
    if samples["PATIENT_ID"].duplicated().any():
        raise ValueError("Multiple samples per patient require an explicit selection rule")
    if not samples["PATIENT_ID"].isin(patients["PATIENT_ID"]).all():
        raise ValueError("Sample table contains patients absent from clinical table")

    with duckdb.connect() as connection:
        connection.register("patients", patients)
        connection.register("samples", samples)
        query = files("surgisense").joinpath("sql/luad_cohort.sql").read_text()
        joined = connection.execute(query).df()

    audit = {
        "patient_rows": len(patients),
        "sample_rows": len(samples),
        "primary_luad_joined": len(joined),
    }
    if joined.empty or joined["patient_id"].duplicated().any():
        raise ValueError("Cohort join must produce unique patient rows")

    status = joined["os_status_raw"].map({"0:LIVING": 0, "1:DECEASED": 1})
    duration = pd.to_numeric(joined["os_months_raw"], errors="coerce")
    valid = status.notna() & np.isfinite(duration) & (duration > 0)
    audit["excluded_missing_or_invalid_survival"] = int((~valid).sum())
    cohort = joined.loc[valid].copy()
    cohort["event"] = status.loc[valid].astype(int).to_numpy()
    cohort["duration_months"] = duration.loc[valid].astype(float).to_numpy()
    cohort["age"] = pd.to_numeric(cohort["age_raw"], errors="coerce")
    cohort.loc[~cohort["age"].between(18, 110), "age"] = np.nan
    cohort["sex"] = cohort["sex_raw"].where(cohort["sex_raw"].isin(["Female", "Male"]))
    cohort["stage"] = cohort["stage_raw"].map(_stage)
    cohort["tmb"] = pd.to_numeric(cohort["tmb_raw"], errors="coerce")
    cohort.loc[cohort["tmb"] < 0, "tmb"] = np.nan
    cohort = cohort[
        ["patient_id", "sample_id", "duration_months", "event", "age", "sex", "stage", "tmb"]
    ].reset_index(drop=True)
    audit["eligible_patients"] = len(cohort)
    audit["observed_deaths"] = int(cohort["event"].sum())
    audit["censored_patients"] = len(cohort) - audit["observed_deaths"]
    for column in ("age", "sex", "stage", "tmb"):
        audit[f"missing_{column}"] = int(cohort[column].isna().sum())
    return cohort, audit
