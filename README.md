# SurgiSense

An open research workflow for constructing a lung adenocarcinoma cohort and evaluating interpretable survival models. The first release compares a clinical baseline (age, sex, pathologic stage) with the same model plus tumor mutational burden (TMB). It uses public TCGA PanCancer Atlas data curated by cBioPortal.

The project demonstrates cohort definition, source verification, SQL, leakage-safe preprocessing, censored survival analysis, model interpretation, and reproducible evaluation. It is research software, not a validated clinical tool.

## Research question

Does adding TMB to simple clinical variables improve out-of-sample discrimination of overall survival in this TCGA LUAD cohort? The comparison is exploratory. It does not establish treatment benefit, clinical utility, or causal effects.

## Dataset and cohort

- Study: [TCGA Lung Adenocarcinoma, PanCancer Atlas 2018](https://www.cbioportal.org/study/summary?id=luad_tcga_pan_can_atlas_2018).
- Source: [cBioPortal DataHub](https://github.com/cBioPortal/datahub/tree/master/public/luad_tcga_pan_can_atlas_2018), patient and sample clinical tables. Downloads are pinned to a DataHub commit and checked by SHA-256 in [`sources.py`](src/surgisense/sources.py).
- Unit: one patient with one primary tumor sample. Duplicate patients or samples fail validation.
- Outcome: overall survival months from initial diagnosis; death is the event, living is right-censored. Records with missing, invalid, or zero follow-up time are excluded. Other missing features are imputed within each training fold.
- Predictors: age at diagnosis, sex, AJCC pathologic stage group, and nonsynonymous TMB. Stage substages are collapsed to I, II, III, IV. TMB is transformed with `log1p` and standardized inside each training fold.
- Cohort SQL: [`luad_cohort.sql`](src/surgisense/sql/luad_cohort.sql).

Raw data and generated reports are kept out of Git. The pipeline records cohort counts, missingness, source checksums, development cross-validation, held-out results, and model coefficients.

## Run

Python 3.12 is supported. Run from the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/surgisense run
.venv/bin/pytest -q
.venv/bin/ruff check src tests
```

`surgisense fetch`, `surgisense cohort`, and `surgisense evaluate` run individual stages. Use `--raw-dir` and `--output-dir` to change their locations. The default inputs are in `data/raw/`; outputs are in `reports/luad/`.

## Evaluation design

Both model specifications are fixed in advance. Patients are split once into 80% development and 20% held-out test sets, stratified by death indicator. Each model receives five-fold stratified cross-validation on development patients. Imputation, one-hot encoding, scaling, and Cox fitting occur independently inside each fold. Final models are fit on all development patients and evaluated once on the same held-out patients. The metric is Harrell's concordance index; 95% bootstrap intervals resample held-out patients and are conditional on the fitted models.

Coefficients are reported as hazard ratios. Age and `log1p(TMB)` coefficients represent a one-standard-deviation change on the development cohort. They are model associations, not causal effects. Stage indicators use stage I as reference; male sex uses female as reference.

## Reproduced result

The pinned source snapshot yields 501 eligible patients, including 181 observed deaths and 320 censored observations. The held-out set contains 101 patients and 36 deaths.

| Prespecified model | Mean 5-fold development C-index | Held-out C-index | Held-out bootstrap 95% interval |
| --- | ---: | ---: | ---: |
| Clinical (age, sex, stage) | 0.688 | 0.567 | 0.457–0.684 |
| Clinical + TMB | 0.688 | 0.557 | 0.446–0.677 |

TMB did not improve discrimination in this split. The held-out intervals are wide, so this result is best read as a demonstration of the analysis workflow, not a biomarker conclusion. Running `surgisense evaluate` regenerates the exact metrics and coefficient tables locally.

## Limits and next work

This is one retrospective TCGA cohort with limited sample size and incomplete follow-up. Pathologic stage and TMB may not be available at the time of initial diagnosis; the analysis makes no real-time risk prediction claim. Missingness, selection into sequencing, treatment, and changing staging editions can bias estimates. Harrell's concordance does not establish calibration or clinical usefulness. External validation and time-specific calibration are needed before stronger claims.

The next increment will add a prespecified RNA expression analysis and an external cohort where comparable variables and outcomes exist. Feature selection must remain inside the training folds. The old postoperative notebooks are retained under `legacy/` as historical experiments and are not part of this oncology workflow.

## Repository layout

```text
src/surgisense/       Download verification, cohort builder, models, CLI
src/surgisense/sql/   Cohort SQL
tests/                Data contract and modeling tests
legacy/               Historical postoperative experiments
data/raw/             Downloaded inputs (ignored)
reports/luad/         Generated outputs (ignored)
```

## License

Project code is MIT licensed. The TCGA source data is provided through cBioPortal under its own data use terms; this repository downloads it but does not redistribute or relicense it. The historical postoperative data is attributed separately in `legacy/postoperative/README.md`.
