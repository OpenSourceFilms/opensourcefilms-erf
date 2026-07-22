"""Atomic file I/O primitives for ERF.

Single-writer rule: every file write uses write-tmp + atomic rename.
MooseFS is POSIX-ish; os.rename() is atomic on the same mount point.
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Union


def write_tmp_rename(path: Union[str, Path], data: bytes) -> None:
    """Write data to a temporary file in the same directory, then atomically rename to path.

    Guarantees: a reader either sees the old file or the new file, never a partial write.
    Zero-byte writes are prohibited — raises ValueError.
    """
    if not data:
        raise ValueError(f"Refusing to write zero bytes to {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.rename(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def write_json(path: Union[str, Path], record: Any, *, indent: int | None = None) -> None:
    """Serialise record to JSON and write atomically."""
    data = json.dumps(record, ensure_ascii=False, indent=indent).encode("utf-8")
    write_tmp_rename(path, data)


def jsonl_append(path: Union[str, Path], record: dict[str, Any]) -> None:
    """Append a single JSON line to a JSONL file, using an exclusive flock for safety.

    The file is created if it doesn't exist. The flock prevents concurrent appends
    from the same process or other processes that also use this function.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    encoded = line.encode("utf-8")
    with open(path, "ab") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.write(encoded)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def read_jsonl(path: Union[str, Path]) -> list[dict[str, Any]]:
    """Read all records from a JSONL file. Returns empty list if file absent."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
