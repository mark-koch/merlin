"""Versioned benchmark result documents and atomic checkpoints."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any


SCHEMA_VERSION = 1


def new_document(*, kind: str, configuration: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "results": [],
    }


def load_document(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported benchmark result schema in {path}")
    if not isinstance(document.get("results"), list):
        raise ValueError(f"benchmark result document has no results list: {path}")
    return document


def atomic_write(path: str | os.PathLike[str], document: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def result_key(row: dict[str, Any]) -> str:
    raw_configuration = row.get("configuration", {})
    configuration = dict(raw_configuration) if isinstance(raw_configuration, dict) else raw_configuration
    if isinstance(configuration, dict):
        configuration.pop("timeout_s", None)
    identity = {
        "case": row.get("case", {}),
        "simulator": row.get("simulator", {}).get("name"),
        "configuration": configuration,
    }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def find_result(document: dict[str, Any], prototype: dict[str, Any]) -> dict[str, Any] | None:
    key = result_key(prototype)
    return next((row for row in document["results"] if result_key(row) == key), None)


def upsert_result(document: dict[str, Any], row: dict[str, Any]) -> None:
    key = result_key(row)
    for index, existing in enumerate(document["results"]):
        if result_key(existing) == key:
            document["results"][index] = row
            return
    document["results"].append(row)
