"""Circuit inspection and checkpoint storage for the cultivation experiment."""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 1


def inspect_circuit(circuit):
    from merlin import CircuitSampler

    source = (ROOT / f"circuit_d3_{circuit}.stim").read_text()
    probabilities = []
    noiseless_gates = {
        "QUBIT_COORDS", "TICK", "SHIFT_COORDS", "R", "RX", "CX",
        "T", "T_DAG", "S", "S_DAG", "DETECTOR", "OBSERVABLE_INCLUDE",
    }
    for line in source.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Z_0-9]+)(?:\(([^)]*)\))?(?:\s+(.*))?", line)
        if match is None:
            raise ValueError(f"Unsupported circuit syntax: {line}")
        gate, argument, targets = match.groups()
        targets = (targets or "").split()
        if gate in {"X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"}:
            width = 2 if gate == "DEPOLARIZE2" else 1
            if len(targets) % width or argument is None:
                raise ValueError(f"Invalid noise instruction: {line}")
            probabilities.extend([float(argument)] * (len(targets) // width))
        elif gate in {"M", "MX", "MPP"}:
            if argument is not None:
                probabilities.extend([float(argument)] * len(targets))
        elif gate not in noiseless_gates:
            raise ValueError(f"Unsupported instruction: {gate}")
    if not probabilities or any(
        not math.isfinite(p) or not 0 < p < 1 or p != probabilities[0]
        for p in probabilities
    ):
        raise ValueError("Expected uniform noise-site probabilities strictly between 0 and 1")
    sampler = CircuitSampler(source, seed=0)
    if sampler.num_fault_sites != len(probabilities):
        raise ValueError("Noise-site count disagrees with merlin")
    metadata = {
        "circuit": circuit,
        "circuit_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "num_fault_sites": sampler.num_fault_sites,
        "num_detectors": sampler.num_detectors,
        "num_observables": sampler.num_observables,
        "num_qubits": sampler.num_qubits,
        "site_probability": probabilities[0],
    }
    return source, metadata


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise ValueError("Expected a positive integer")
    return value


def validate_checkpoint(data):
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema version")
    metadata = data["metadata"]
    config = data["config"]
    if metadata["circuit"] not in {"t", "s"}:
        raise ValueError("Unknown circuit")
    for value in (metadata["num_fault_sites"], config["max_num_faults"],
                  config["batch_size"], config["shots_per_num_faults"]):
        if type(value) is not int or value <= 0:
            raise ValueError("Invalid checkpoint configuration")
    if config["max_num_faults"] > metadata["num_fault_sites"]:
        raise ValueError("Maximum num_faults exceeds number of sites")
    groups = {num_faults: [] for num_faults in range(1, config["max_num_faults"] + 1)}
    for batch in data["batches"]:
        fields = [batch[name] for name in ("num_faults", "offset", "total", "accepted", "errors")]
        if any(type(value) is not int for value in fields):
            raise ValueError("Batch counts must be integers")
        num_faults, offset, total, accepted, errors = fields
        if (num_faults not in groups or offset < 0 or not 0 <= errors <= accepted <= total
                or not 0 < total <= config["batch_size"]
                or offset + total > config["shots_per_num_faults"]):
            raise ValueError("Invalid batch counts or range")
        groups[num_faults].append(batch)
    for batches in groups.values():
        end = 0
        for batch in sorted(batches, key=lambda item: item["offset"]):
            if batch["offset"] < end:
                raise ValueError("Overlapping or duplicate checkpoint batches")
            end = batch["offset"] + batch["total"]
    return groups


def load_checkpoint(path):
    try:
        data = json.loads(Path(path).read_text())
        validate_checkpoint(data)
        return data
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid checkpoint {path}: {error}") from error


def save_checkpoint(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data["batches"].sort(key=lambda batch: (batch["num_faults"], batch["offset"]))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = stream.name
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
