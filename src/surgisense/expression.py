"""Join RNA abundance to clinical patients without guessing sample or gene identities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from surgisense.sources import RNA_SOURCE, verify_sources


def load_expression(
    raw_dir: Path, cohort: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, list[str], dict[str, int]]:
    """Return matched patients and log2(RSEM + 1) genes in the same row order."""
    path = verify_sources(raw_dir, include_rna=True)[next(iter(RNA_SOURCE))]
    with path.open() as source:
        header = source.readline().rstrip("\r\n").split("\t")
    if header[:2] != ["Hugo_Symbol", "Entrez_Gene_Id"]:
        raise ValueError("Unexpected RNA file schema")
    sample_ids = header[2:]
    if not sample_ids or len(sample_ids) != len(set(sample_ids)) or any(not s for s in sample_ids):
        raise ValueError("RNA file has missing or duplicate sample IDs")
    if cohort["patient_id"].duplicated().any() or cohort["sample_id"].duplicated().any():
        raise ValueError("Clinical cohort must have unique patients and samples")

    matched = cohort.loc[cohort["sample_id"].isin(sample_ids)].reset_index(drop=True)
    if matched.empty:
        raise ValueError("No clinical patients have RNA measurements")
    selected_samples = matched["sample_id"].tolist()
    table = pd.read_csv(
        path,
        sep="\t",
        usecols=["Entrez_Gene_Id", *selected_samples],
        dtype={"Entrez_Gene_Id": "string"},
    )
    gene_ids = table["Entrez_Gene_Id"]
    ambiguous = gene_ids.isna() | gene_ids.eq("") | gene_ids.duplicated(keep=False)
    table = table.loc[~ambiguous]
    if table.empty:
        raise ValueError("RNA file has no unambiguous gene IDs")
    values = table[selected_samples].to_numpy(dtype=np.float32).T
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("RNA abundance must be finite and nonnegative")
    expression = np.log2(values + 1.0)
    audit = {
        "clinical_patients": len(cohort),
        "rna_source_samples": len(sample_ids),
        "matched_patients": len(matched),
        "excluded_without_rna": len(cohort) - len(matched),
        "rna_source_gene_rows": len(gene_ids),
        "excluded_ambiguous_gene_rows": int(ambiguous.sum()),
        "retained_unique_genes": len(table),
    }
    return matched, expression, table["Entrez_Gene_Id"].tolist(), audit
