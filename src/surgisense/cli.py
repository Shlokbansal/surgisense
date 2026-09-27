"""Command-line workflow for fetching, building, and evaluating the LUAD cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from surgisense.cohort import build_cohort
from surgisense.expression import load_expression
from surgisense.modeling import evaluate
from surgisense.rna_modeling import evaluate_rna
from surgisense.sources import DATAHUB_COMMIT, RNA_SOURCE, SOURCES, fetch_sources


def main() -> None:
    parser = argparse.ArgumentParser(description="SurgiSense LUAD research workflow")
    parser.add_argument(
        "command", choices=["fetch", "cohort", "evaluate", "run", "fetch-rna", "evaluate-rna"]
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/luad"))
    args = parser.parse_args()

    if args.command in ("fetch", "run", "fetch-rna"):
        fetched = fetch_sources(args.raw_dir, include_rna=args.command == "fetch-rna")
        print(f"Verified {len(fetched)} source files in {args.raw_dir}")
    if args.command in ("fetch", "fetch-rna"):
        return

    cohort, audit = build_cohort(args.raw_dir)
    if args.command == "evaluate-rna":
        matched, expression, gene_ids, rna_audit = load_expression(args.raw_dir, cohort)
        rna_report, rna_coefficients = evaluate_rna(matched, expression, gene_ids)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "rna_cohort_audit.json").write_text(
            json.dumps({
                "study": "luad_tcga_pan_can_atlas_2018",
                "datahub_commit": DATAHUB_COMMIT,
                "sources_sha256": SOURCES | RNA_SOURCE,
                **rna_audit,
            }, indent=2) + "\n"
        )
        (args.output_dir / "rna_evaluation.json").write_text(
            json.dumps(rna_report, indent=2) + "\n"
        )
        rna_coefficients.to_csv(args.output_dir / "rna_coefficients.csv", index=False)
        print(f"Compared models on {len(matched)} patients with tumor RNA")
        print(json.dumps({
            "models": rna_report["models"],
            "held_out_rna_minus_clinical_c_index": (
                rna_report["held_out_rna_minus_clinical_c_index"]
            ),
            "fixed_horizon_validation": rna_report["fixed_horizon_validation"],
        }, indent=2))
        return

    report = coefficients = None
    if args.command in ("evaluate", "run"):
        report, coefficients = evaluate(cohort)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(args.output_dir / "cohort.parquet", index=False)
    (args.output_dir / "cohort_audit.json").write_text(
        json.dumps(
            {
                "study": "luad_tcga_pan_can_atlas_2018",
                "datahub_commit": DATAHUB_COMMIT,
                "sources_sha256": SOURCES,
                **audit,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Built cohort with {len(cohort)} patients and {audit['observed_deaths']} deaths")
    if args.command == "cohort":
        return

    assert report is not None and coefficients is not None
    (args.output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, table in coefficients.items():
        table.to_csv(args.output_dir / f"{name}_coefficients.csv", index=False)
    print(json.dumps({
        "models": report["models"],
        "fixed_horizon_validation": report["fixed_horizon_validation"],
    }, indent=2))


if __name__ == "__main__":
    main()
