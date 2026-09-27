"""Command-line workflow for fetching, building, and evaluating the LUAD cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from surgisense.cohort import build_cohort
from surgisense.modeling import evaluate
from surgisense.sources import DATAHUB_COMMIT, SOURCES, fetch_sources


def main() -> None:
    parser = argparse.ArgumentParser(description="SurgiSense LUAD research workflow")
    parser.add_argument("command", choices=["fetch", "cohort", "evaluate", "run"])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/luad"))
    args = parser.parse_args()

    if args.command in ("fetch", "run"):
        fetched = fetch_sources(args.raw_dir)
        print(f"Verified {len(fetched)} source files in {args.raw_dir}")
    if args.command == "fetch":
        return

    cohort, audit = build_cohort(args.raw_dir)
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
    print(json.dumps(report["models"], indent=2))


if __name__ == "__main__":
    main()
