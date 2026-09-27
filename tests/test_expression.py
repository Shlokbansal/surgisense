from pathlib import Path

import pandas as pd
import pytest

from surgisense import expression


def test_expression_join_preserves_patient_order_and_drops_ambiguous_genes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "rna.txt"
    path.write_text(
        "Hugo_Symbol\tEntrez_Gene_Id\tS2\tS1\n"
        "A\t101\t3\t1\n"
        "B\t102\t7\t2\n"
        "C\t102\t8\t2\n"
        "D\t103\t0\t0\n"
    )
    monkeypatch.setattr(expression, "verify_sources", lambda *_args, **_kwargs: {
        "data_mrna_seq_v2_rsem.txt": path
    })
    cohort = pd.DataFrame({
        "patient_id": ["P1", "P2", "P3"], "sample_id": ["S1", "S2", "S3"]
    })
    matched, values, genes, audit = expression.load_expression(tmp_path, cohort)
    assert matched["patient_id"].tolist() == ["P1", "P2"]
    assert genes == ["101", "103"]
    assert values.shape == (2, 2)
    assert values[0, 0] == 1.0  # log2(1 + 1)
    assert values[1, 0] == 2.0  # log2(3 + 1)
    assert audit["excluded_without_rna"] == 1
    assert audit["excluded_ambiguous_gene_rows"] == 2


def test_expression_rejects_duplicate_sample_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "rna.txt"
    path.write_text("Hugo_Symbol\tEntrez_Gene_Id\tS1\tS1\nA\t101\t1\t1\n")
    monkeypatch.setattr(expression, "verify_sources", lambda *_args, **_kwargs: {
        "data_mrna_seq_v2_rsem.txt": path
    })
    cohort = pd.DataFrame({"patient_id": ["P1"], "sample_id": ["S1"]})
    with pytest.raises(ValueError, match="duplicate sample IDs"):
        expression.load_expression(tmp_path, cohort)
