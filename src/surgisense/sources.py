"""Versioned public inputs for the LUAD research cohort."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen

STUDY = "luad_tcga_pan_can_atlas_2018"
DATAHUB_COMMIT = "8119e63a2c0c4fb1971cd67312a9a4e7636b825d"
BASE_URL = (
    f"https://media.githubusercontent.com/media/cBioPortal/datahub/{DATAHUB_COMMIT}/"
    f"public/{STUDY}"
)
SOURCES = {
    "data_clinical_patient.txt": "2c347251474ec48a365e9b9dcd3b86a7106a6bbf69469ec4b740ba4d92695df7",
    "data_clinical_sample.txt": "b443896a6aed8256f3aa1b5e2f41e700392c76461be10936d2c344988bc5d84e",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources(raw_dir: Path) -> dict[str, Path]:
    paths = {name: raw_dir / name for name in SOURCES}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run 'surgisense fetch' first")
        actual = sha256(path)
        if actual != SOURCES[name]:
            raise ValueError(f"Checksum mismatch for {path}: {actual}")
    return paths


def fetch_sources(raw_dir: Path) -> dict[str, Path]:
    """Download each source atomically and reject changed upstream files."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in SOURCES.items():
        destination = raw_dir / name
        if destination.is_file() and sha256(destination) == expected:
            continue
        descriptor, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=raw_dir)
        try:
            with (
                os.fdopen(descriptor, "wb") as output,
                urlopen(f"{BASE_URL}/{name}", timeout=60) as response,
            ):
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if sha256(Path(temporary)) != expected:
                raise ValueError(f"Upstream {name} changed; review and update its pinned checksum")
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return verify_sources(raw_dir)
