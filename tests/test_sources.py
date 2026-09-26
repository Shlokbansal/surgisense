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
