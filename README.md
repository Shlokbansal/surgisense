# SurgiSense

An open research workflow for constructing a lung adenocarcinoma cohort and evaluating survival models. The clinical baseline (age, sex, pathologic stage) is compared separately with tumor mutational burden (TMB) and tumor RNA abundance. It uses public TCGA PanCancer Atlas data curated by cBioPortal.

The project demonstrates cohort definition, source verification, SQL, leakage-safe preprocessing, censored survival analysis, model interpretation, and reproducible evaluation. It is research software, not a validated clinical tool.

## Research question

Do TMB or tumor RNA measurements add out-of-sample discrimination of overall survival beyond simple clinical variables in this TCGA LUAD cohort? The comparisons are exploratory. They do not establish treatment benefit, clinical utility, or causal effects.

## Dataset and cohort

- Study: [TCGA Lung Adenocarcinoma, PanCancer Atlas 2018](https://www.cbioportal.org/study/summary?id=luad_tcga_pan_can_atlas_2018).
- Source: [cBioPortal DataHub](https://github.com/cBioPortal/datahub/tree/master/public/luad_tcga_pan_can_atlas_2018), patient and sample clinical tables plus the RNA-Seq V2 RSEM abundance matrix. Downloads are pinned to a DataHub commit and checked by SHA-256 in [`sources.py`](src/surgisense/sources.py).
- Unit: one patient with one primary tumor sample. Duplicate patients or samples fail validation.
- Outcome: overall survival months from initial diagnosis; death is the event, living is right-censored. Records with missing, invalid, or zero follow-up time are excluded. Other missing features are imputed within each training fold.
- Predictors: age at diagnosis, sex, AJCC pathologic stage group, and nonsynonymous TMB. Stage substages are collapsed to I, II, III, IV. TMB is transformed with `log1p` and standardized inside each training fold.
- Cohort SQL: [`luad_cohort.sql`](src/surgisense/sql/luad_cohort.sql).

Raw data and generated reports are kept out of Git. The pipeline records cohort counts, missingness, source checksums, development cross-validation, held-out results, and model coefficients. RNA is an optional 73 MB download; the clinical/TMB workflow does not require it.

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

To run the separate tumor RNA experiment:

```bash
.venv/bin/surgisense fetch-rna
.venv/bin/surgisense evaluate-rna
```

This writes `rna_cohort_audit.json`, `rna_evaluation.json`, and `rna_coefficients.csv` under `reports/luad/`. It does not alter the clinical/TMB evaluation.

## Evaluation design

Both model specifications are fixed in advance. Patients are split once into 80% development and 20% held-out test sets, stratified by death indicator. Each model receives five-fold stratified cross-validation on development patients. Imputation, one-hot encoding, scaling, and Cox fitting occur independently inside each fold. Final models are fit on all development patients and evaluated once on the same held-out patients. The metric is Harrell's concordance index; 95% bootstrap intervals resample held-out patients and are conditional on the fitted models.

Coefficients are reported as hazard ratios. Age and `log1p(TMB)` coefficients represent a one-standard-deviation change on the development cohort. They are model associations, not causal effects. Stage indicators use stage I as reference; male sex uses female as reference.

The generated clinical/TMB evaluation includes development-only Schoenfeld-residual checks of the proportional-hazards assumption. These p-values are exploratory and unadjusted for multiple comparisons.

Both experiments also evaluate absolute survival probabilities at prespecified 12- and 24-month horizons. The [inverse-probability-of-censoring-weighted Brier score](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.brier_score.html) measures prediction error when some patients leave follow-up before a horizon; lower is better. The censoring curve and a simple constant-probability reference model are fit on development patients only. The held-out Kaplan–Meier estimate is used only for a descriptive comparison of the average predicted and observed survival rates. This is a population-level check, **not** proof that individual probabilities are calibrated. The marginal censoring model assumes censoring is independent of survival; informative follow-up could invalidate it.

The RNA analysis first matches the expression matrix to patients by exact primary-tumor sample ID. It drops gene rows with missing or repeated Entrez identifiers rather than guessing which row to keep. It applies the fixed transform `log2(RSEM + 1)`; within each training fold, it selects the 1,000 most variable genes, standardizes them, and compresses them to five principal components (PCs). Gene selection, scaling, PCA, clinical preprocessing, and Cox fitting are all fit inside each training fold. The fixed RNA model is compared with a clinical-only model on exactly the same RNA-matched patients and held-out set. A paired bootstrap interval resamples the same held-out patients for the difference in C-index. PCs are broad mathematical summaries, not identified biomarkers. The RNA report also includes development-only exploratory proportional-hazards checks.

## Reproduced result

The pinned source snapshot yields 501 eligible patients, including 181 observed deaths and 320 censored observations. The held-out set contains 101 patients and 36 deaths.

| Prespecified model | Mean 5-fold development C-index | Held-out C-index | Held-out bootstrap 95% interval |
| --- | ---: | ---: | ---: |
| Clinical (age, sex, stage) | 0.688 | 0.567 | 0.457–0.684 |
| Clinical + TMB | 0.688 | 0.557 | 0.446–0.677 |

TMB did not improve discrimination in this split. The held-out intervals are wide, so this result is best read as a demonstration of the analysis workflow, not a biomarker conclusion. Running `surgisense evaluate` regenerates the exact metrics and coefficient tables locally.

The proportional-hazards check for sex is approximately p=0.027 in both models. A fixed hazard ratio for sex may therefore be an oversimplification; the coefficient table is descriptive and should not be interpreted as a stable effect over follow-up.

### RNA experiment

Of the 501 eligible patients, 497 have an exact RNA sample match; four are excluded from both arms of this comparison. The source has 20,531 gene rows; 50 rows with repeated Entrez IDs are dropped, leaving 20,481 unambiguous genes. The 497 matched patients are split into 397 development and 100 held-out patients.

| Prespecified model on RNA-matched patients | Mean 5-fold development C-index | Held-out C-index | Held-out bootstrap 95% interval |
| --- | ---: | ---: | ---: |
| Clinical (age, sex, stage) | 0.694 | 0.515 | 0.405–0.634 |
| Clinical + RNA PCs | 0.722 | 0.455 | 0.330–0.580 |

The held-out difference (RNA minus clinical) is −0.061, with a paired bootstrap 95% interval of −0.121 to 0.007. RNA improved development cross-validation but **did not improve held-out discrimination** in this fixed analysis. This is consistent with possible overfitting and is not evidence that RNA lacks prognostic value generally. The RNA specification was not revised after looking at the held-out result. The RNA-matched split is different from the earlier 501-patient TMB split, so the two tables should not be compared row-for-row.

### Fixed-horizon probability checks

The earlier C-index asks whether patients are ranked in roughly the right order. This check asks a different question: are the predicted chances of surviving one or two years accurate? A model can do acceptably on one question and poorly on the other.

The constant reference assigns everyone the same survival probability estimated from development patients; it does not use patient features. Brier scores below are held-out errors, so smaller is better. Scores from the 501-patient and 497-patient experiments use different held-out sets and should not be compared directly.

| Experiment | Horizon | Constant reference | Clinical | Clinical + molecular data |
| --- | ---: | ---: | ---: | ---: |
| Clinical / TMB (501 patients) | 12 months | 0.088 | 0.089 | TMB: 0.089 |
| Clinical / TMB (501 patients) | 24 months | 0.168 | 0.169 | TMB: 0.171 |
| RNA-matched (497 patients) | 12 months | 0.080 | 0.082 | RNA: 0.084 |
| RNA-matched (497 patients) | 24 months | 0.170 | 0.182 | RNA: 0.197 |

No fitted model beat the constant reference in these point estimates. On the original clinical/TMB held-out set, the clinical model predicted an average 24-month death probability of 25.1%, while the held-out Kaplan–Meier estimate was 18.9% (17 deaths observed by 24 months among 101 held-out patients). These small samples and incomplete follow-up make the gaps uncertain; this is a warning against presenting the outputs as reliable patient-level risk estimates, not a definitive comparison of all possible models. The full unrounded values and censoring counts are in the generated `evaluation.json` and `rna_evaluation.json` reports.

## Limits and next work

This is one retrospective TCGA cohort with limited sample size and incomplete follow-up. Pathologic stage and TMB may not be available at the time of initial diagnosis; the analysis makes no real-time risk prediction claim. Specimen collection dates are unavailable in these tables, so delayed entry cannot be addressed and selection into sequencing may bias the estimates. Missingness, treatment, and changing staging editions are additional concerns. Harrell's concordance alone does not establish calibration or clinical usefulness, and the new fixed-horizon checks show weak probability performance. External validation and fuller calibration assessment are needed before stronger claims.

The next increment should seek an external cohort where comparable variables and outcomes exist. Any near-term public MVP should explain the cohort, methods, and results rather than present an individual patient risk calculator as validated. The old postoperative notebooks are retained under `legacy/` as historical experiments and are not part of this oncology workflow.

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
