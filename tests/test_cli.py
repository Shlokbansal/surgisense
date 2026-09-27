import sys
from pathlib import Path

import pandas as pd
import pytest

from surgisense import cli


def test_failed_evaluation_does_not_write_partial_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "report"
    monkeypatch.setattr(cli, "build_cohort", lambda _: (pd.DataFrame(), {}))

    def fail_evaluation(_: pd.DataFrame) -> None:
        raise RuntimeError("model failed")

    monkeypatch.setattr(cli, "evaluate", fail_evaluation)
    monkeypatch.setattr(
        sys,
        "argv",
        ["surgisense", "evaluate", "--raw-dir", str(tmp_path), "--output-dir", str(output)],
    )
    with pytest.raises(RuntimeError, match="model failed"):
        cli.main()
    assert not output.exists()
