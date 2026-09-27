"""Export only aggregate, public-safe research results for the static dashboard."""

from __future__ import annotations

import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run both evaluation commands first")
    return json.loads(path.read_text())


def _bounded(value: float, label: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise ValueError(f"{label} must be between zero and one")
    return number


def _experiment(report: dict, names: tuple[str, str], labels: tuple[str, str]) -> dict:
    models = []
    for name, label in zip(names, labels, strict=True):
        model = report["models"][name]
        interval = model["held_out_bootstrap_95pct_interval"]
        if len(interval) != 2 or interval[0] > interval[1]:
            raise ValueError(f"{name} C-index interval is invalid")
        models.append({
            "key": name,
            "label": label,
            "development_cv_c_index": _bounded(
                model["development_cv_mean_c_index"], f"{name} development C-index"
            ),
            "held_out_c_index": _bounded(model["held_out_c_index"], f"{name} C-index"),
            "held_out_c_index_interval": [
                _bounded(value, f"{name} C-index interval")
                for value in interval
            ],
        })

    horizons = []
    for month in (12, 24):
        source = report["fixed_horizon_validation"][str(month)]
        if source["held_out_deaths_by_horizon"] > report["held_out_deaths"]:
            raise ValueError("Horizon deaths exceed held-out deaths")
        horizons.append({
            "months": month,
            "observed_death_probability": _bounded(
                source["held_out_observed_km_death"], "observed death probability"
            ),
            "held_out_deaths": int(source["held_out_deaths_by_horizon"]),
            "held_out_censored": int(source["held_out_censored_by_horizon"]),
            "constant_reference_brier": _bounded(
                source["development_km_constant_ipcw_brier_score"], "reference Brier"
            ),
            "models": [
                {
                    "key": name,
                    "brier": _bounded(source["models"][name]["ipcw_brier_score"], "Brier"),
                    "mean_predicted_death": _bounded(
                        source["models"][name]["mean_predicted_death"], "predicted death"
                    ),
                }
                for name in names
            ],
        })
    return {
        "development_patients": int(report["development_patients"]),
        "held_out_patients": int(report["held_out_patients"]),
        "development_deaths": int(report["development_deaths"]),
        "held_out_deaths": int(report["held_out_deaths"]),
        "models": models,
        "horizons": horizons,
    }


def build_public_summary(report_dir: Path) -> dict:
    """Whitelist aggregate fields; never copy full reports with patient IDs."""
    audit = _read_json(report_dir / "cohort_audit.json")
    rna_audit = _read_json(report_dir / "rna_cohort_audit.json")
    clinical = _read_json(report_dir / "evaluation.json")
    rna = _read_json(report_dir / "rna_evaluation.json")
    if audit["datahub_commit"] != rna_audit["datahub_commit"]:
        raise ValueError("Clinical and RNA reports have different source snapshots")
    if audit["eligible_patients"] != (
        clinical["development_patients"] + clinical["held_out_patients"]
    ) or audit["observed_deaths"] != (
        clinical["development_deaths"] + clinical["held_out_deaths"]
    ):
        raise ValueError("Clinical report and cohort counts disagree")
    if rna_audit["matched_patients"] != (
        rna["development_patients"] + rna["held_out_patients"]
    ) or rna_audit["clinical_patients"] != audit["eligible_patients"]:
        raise ValueError("RNA report and cohort counts disagree")
    if rna_audit["matched_patients"] + rna_audit["excluded_without_rna"] != (
        audit["eligible_patients"]
    ) or rna_audit["retained_unique_genes"] + (
        rna_audit["excluded_ambiguous_gene_rows"]
    ) != rna_audit["rna_source_gene_rows"]:
        raise ValueError("RNA source and exclusion counts disagree")

    return {
        "schema_version": 1,
        "study": "TCGA Lung Adenocarcinoma, PanCancer Atlas 2018",
        "datahub_commit": audit["datahub_commit"],
        "cohort": {
            "eligible_patients": int(audit["eligible_patients"]),
            "observed_deaths": int(audit["observed_deaths"]),
            "censored_patients": int(audit["censored_patients"]),
            "missing_features": {
                name: int(audit[f"missing_{name}"])
                for name in ("age", "sex", "stage", "tmb")
            },
        },
        "rna_cohort": {
            "matched_patients": int(rna_audit["matched_patients"]),
            "excluded_without_rna": int(rna_audit["excluded_without_rna"]),
            "source_gene_rows": int(rna_audit["rna_source_gene_rows"]),
            "excluded_ambiguous_gene_rows": int(
                rna_audit["excluded_ambiguous_gene_rows"]
            ),
            "retained_unique_genes": int(rna_audit["retained_unique_genes"]),
        },
        "experiments": {
            "clinical_tmb": _experiment(
                clinical,
                ("clinical", "clinical_plus_tmb"),
                ("Clinical", "Clinical + TMB"),
            ),
            "rna": _experiment(
                rna,
                ("clinical", "clinical_plus_rna"),
                ("Clinical", "Clinical + RNA"),
            ),
        },
    }


def export_public_summary(report_dir: Path, output_path: Path) -> dict:
    summary = build_public_summary(report_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary
