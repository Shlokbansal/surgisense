# Historical postoperative experiments

These files are retained for provenance only. The original [90-row UCI dataset](https://archive.ics.uci.edu/dataset/82/post%2Boperative%2Bpatient) predicts postoperative discharge destination (ICU, home, or general hospital floor), not complications. The notebooks are exploratory; `eda.py` was previously stored with an `.ipynb` extension but is a Python script. The preprocessing code was built for learning and fits transformations before data splitting, so its model results should not be used as validation evidence.

The current oncology workflow lives in `src/surgisense/` and does not import this directory.

Dataset credit: Summers, S. & Woolery, L. (1991), *Post-Operative Patient*, UCI Machine Learning Repository, [DOI 10.24432/C5DG6Q](https://doi.org/10.24432/C5DG6Q). The dataset is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The CSV in this directory adds column names to the original data; the rows are otherwise the historical UCI dataset.
