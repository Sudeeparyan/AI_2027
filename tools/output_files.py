"""Replace a generated file only after its complete contents have been written."""
from __future__ import annotations

import os
import errno
from pathlib import Path
import tempfile
import time


def atomic_text(path: Path, text: str) -> None:
    """Keep the old file intact; briefly retry transient Windows reader locks."""
    _atomic_write(path, text, "w")


def atomic_copy(source: Path, destination: Path) -> None:
    """Copy complete asset bytes; leave an identical existing asset untouched."""
    source, destination = Path(source), Path(destination)
    content = source.read_bytes()
    if destination.exists() and destination.read_bytes() == content:
        return
    _atomic_write(destination, content, "wb")


def _atomic_write(path: Path, content: str | bytes, mode: str) -> None:
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, mode, encoding="utf-8" if mode == "w" else None) as stream:
            stream.write(content)
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                break
            except OSError as error:
                transient = getattr(error, "winerror", None) in {5, 32, 33, 1224} or error.errno in {errno.EACCES, errno.EBUSY, errno.EPERM}
                if not transient or attempt == 4:
                    raise
                time.sleep(0.1 * 2 ** attempt)
    finally:
        temporary.unlink(missing_ok=True)
