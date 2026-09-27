import json
from pathlib import Path

import pytest

from surgisense.dashboard_export import build_public_summary


def _report(second: str) -> dict:
    model = {
        "development_cv_mean_c_index": 0.6,
        "held_out_c_index": 0.55,
        "held_out_bootstrap_95pct_interval": [0.4, 0.7],
    }
    horizon = {
        "held_out_observed_km_death": 0.2,
        "held_out_deaths_by_horizon": 1,
        "held_out_censored_by_horizon": 1,
        "development_km_constant_ipcw_brier_score": 0.15,
        "models": {
            "clinical": {"ipcw_brier_score": 0.16, "mean_predicted_death": 0.21},
            second: {"ipcw_brier_score": 0.17, "mean_predicted_death": 0.22},
        },
    }
    return {
        "development_patients": 8,
        "held_out_patients": 2,
        "development_deaths": 3,
        "held_out_deaths": 1,
        "split": {"development_patient_ids": ["PRIVATE_PATIENT_ID"]},
        "development_selected_gene_ids": ["PRIVATE_GENE_ID"],
        "models": {"clinical": model, second: model},
        "fixed_horizon_validation": {"12": horizon, "24": horizon},
    }


def _write_reports(path: Path) -> None:
    reports = {
        "cohort_audit.json": {
            "datahub_commit": "test-commit", "eligible_patients": 10,
            "observed_deaths": 4, "censored_patients": 6,
            "missing_age": 0, "missing_sex": 0, "missing_stage": 1, "missing_tmb": 2,
        },
        "rna_cohort_audit.json": {
            "datahub_commit": "test-commit", "clinical_patients": 10,
            "matched_patients": 10, "excluded_without_rna": 0,
            "rna_source_gene_rows": 5, "excluded_ambiguous_gene_rows": 1,
            "retained_unique_genes": 4,
        },
        "evaluation.json": _report("clinical_plus_tmb"),
        "rna_evaluation.json": _report("clinical_plus_rna"),
    }
    for filename, content in reports.items():
        (path / filename).write_text(json.dumps(content))


def test_dashboard_export_whitelists_aggregate_fields(tmp_path: Path) -> None:
    _write_reports(tmp_path)
    summary = build_public_summary(tmp_path)
    serialized = json.dumps(summary)
    assert "PRIVATE_PATIENT_ID" not in serialized
    assert "PRIVATE_GENE_ID" not in serialized
    assert summary["cohort"]["eligible_patients"] == 10
    assert summary["experiments"]["rna"]["horizons"][0]["models"][1]["brier"] == 0.17


def test_dashboard_export_rejects_mismatched_cohorts(tmp_path: Path) -> None:
    _write_reports(tmp_path)
    path = tmp_path / "rna_cohort_audit.json"
    audit = json.loads(path.read_text())
    audit["matched_patients"] = 9
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="RNA report and cohort counts disagree"):
        build_public_summary(tmp_path)
