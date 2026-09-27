from pathlib import Path

import pytest

from surgisense import sources


def test_source_checksum_rejects_changed_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "input.txt"
    path.write_text("original")
    expected = sources.sha256(path)
    monkeypatch.setattr(sources, "SOURCES", {"input.txt": expected})
    assert sources.verify_sources(tmp_path) == {"input.txt": path}
    path.write_text("changed")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        sources.verify_sources(tmp_path)


def test_rna_source_is_optional_but_verified_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clinical = tmp_path / "clinical.txt"
    clinical.write_text("clinical")
    rna = tmp_path / "rna.txt"
    rna.write_text("rna")
    monkeypatch.setattr(sources, "SOURCES", {"clinical.txt": sources.sha256(clinical)})
    monkeypatch.setattr(sources, "RNA_SOURCE", {"rna.txt": sources.sha256(rna)})
    assert sources.verify_sources(tmp_path) == {"clinical.txt": clinical}
    assert sources.verify_sources(tmp_path, include_rna=True) == {
        "clinical.txt": clinical, "rna.txt": rna
    }
    rna.write_text("changed")
    assert sources.verify_sources(tmp_path) == {"clinical.txt": clinical}
    with pytest.raises(ValueError, match="Checksum mismatch"):
        sources.verify_sources(tmp_path, include_rna=True)
