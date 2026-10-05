"""Collect fixed-num_faults cultivation samples with resumable checkpoints."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
from functools import lru_cache
from importlib.metadata import version
import multiprocessing
import os
from pathlib import Path
import sys

if __package__:
    from .common import (ROOT, SCHEMA_VERSION, inspect_circuit, load_checkpoint,
                         positive_int, save_checkpoint, validate_checkpoint)
else:
    from common import (ROOT, SCHEMA_VERSION, inspect_circuit, load_checkpoint,
                        positive_int, save_checkpoint, validate_checkpoint)


@lru_cache(maxsize=2)
def compile_clifft(source, num_detectors, num_observables, num_fault_sites, site_probability):
    import clifft
    import numpy as np

    program = clifft.compile(source, normalize_syndromes=True,
                             postselection_mask=[1] * num_detectors)
    probabilities = np.asarray(program.noise_site_probabilities)
    if (program.num_detectors != num_detectors
            or program.num_observables != num_observables
            or probabilities.shape != (num_fault_sites,)
            or not np.all(probabilities == site_probability)):
        raise ValueError("Clifft compiled metadata disagrees with the circuit metadata")
    return program


def clifft_program(source, metadata):
    return compile_clifft(source, *(metadata[name] for name in (
        "num_detectors", "num_observables", "num_fault_sites", "site_probability")))


def sample_batch(source, metadata, seed, num_faults, offset, shots, *, backend="merlin"):

    identity = f"{seed}:{metadata['circuit_sha256']}:{num_faults}:{offset}"
    batch_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little")
    if backend == "clifft":
        import clifft

        result = clifft.sample_k_survivors(
            clifft_program(source, metadata), shots=shots, k=num_faults,
            seed=batch_seed, threads=1, keep_records=False,
        )
        accepted = result.passed_shots
        errors = result.logical_errors
    elif backend == "merlin":
        from merlin import CircuitSampler

        sampler = CircuitSampler(source, seed=batch_seed,
                                 postselect=list(range(metadata["num_detectors"])))
        result = sampler.sample(shots, num_faults=num_faults)
        accepted = result.accepted_shots
        errors = result.observables.any(axis=1).sum()
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return {
        "num_faults": num_faults, "offset": offset,
        "total": int(result.total_shots), "accepted": int(accepted),
        "errors": int(errors),
    }


def pending_batches(data):
    config = data["config"]
    for num_faults, batches in validate_checkpoint(data).items():
        offset = 0
        for batch in sorted(batches, key=lambda item: item["offset"]) + [
            {"offset": config["shots_per_num_faults"], "total": 0}
        ]:
            while offset < batch["offset"]:
                shots = min(config["batch_size"], batch["offset"] - offset)
                yield num_faults, offset, shots
                offset += shots
            offset = batch["offset"] + batch["total"]


def collect(args):
    source, metadata = inspect_circuit(args.circuit)
    if args.max_num_faults > metadata["num_fault_sites"]:
        raise ValueError("Maximum num_faults exceeds number of sites")
    if not 0 <= args.seed < 2**64:
        raise ValueError("Seed must be an unsigned 64-bit integer")
    config = {name: getattr(args, name) for name in (
        "shots_per_num_faults", "max_num_faults", "batch_size", "seed")}
    backend_name = getattr(args, "backend", "merlin")
    if backend_name not in {"merlin", "clifft"}:
        raise ValueError(f"Unknown backend: {backend_name}")
    backend = {"name": backend_name,
               "version": version("merlin" if backend_name == "merlin" else "clifft")}
    filename = (f"samples_{args.circuit}.json" if backend_name == "merlin"
                else f"samples_clifft_{args.circuit}.json")
    output = args.output or ROOT / filename
    if output.exists():
        data = load_checkpoint(output)
        if data["metadata"] != metadata or data["backend"] != backend:
            raise ValueError("Checkpoint circuit or backend version does not match")
        for name in ("max_num_faults", "batch_size", "seed"):
            if data["config"][name] != config[name]:
                raise ValueError(f"Checkpoint {name} does not match")
        if config["shots_per_num_faults"] < data["config"]["shots_per_num_faults"]:
            raise ValueError("Cannot decrease the checkpoint shot budget")
        data["config"] = config
    else:
        data = {"schema_version": SCHEMA_VERSION, "metadata": metadata,
                "backend": backend, "config": config, "batches": []}
    if backend_name == "clifft":
        clifft_program(source, metadata)
    save_checkpoint(output, data)
    tasks = iter(list(pending_batches(data)))
    completed = sum(batch["total"] for batch in data["batches"])
    target = args.shots_per_num_faults * args.max_num_faults

    def commit(batch):
        nonlocal completed
        data["batches"].append(batch)
        save_checkpoint(output, data)
        completed += batch["total"]
        print(f"{args.circuit.upper()} ({backend_name}): {completed}/{target} attempts; "
              f"num_faults={batch['num_faults']}, accepted={batch['accepted']}, "
              f"errors={batch['errors']}", file=sys.stderr, flush=True)

    if args.workers == 1:
        for task in tasks:
            commit(sample_batch(source, metadata, args.seed, *task, backend=backend_name))
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers,
                                   mp_context=multiprocessing.get_context("spawn"))
        pending = set()
        try:
            while True:
                while len(pending) < 2 * args.workers:
                    task = next(tasks, None)
                    if task is None:
                        break
                    pending.add(pool.submit(sample_batch, source, metadata, args.seed,
                                            *task, backend=backend_name))
                if not pending:
                    break
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    commit(future.result())
        finally:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
    print(f"Checkpoint: {output} ({completed}/{target} attempts)", file=sys.stderr)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit", choices=("t", "s"), required=True)
    parser.add_argument("--backend", choices=("merlin", "clifft"), default="merlin")
    parser.add_argument("--shots-per-num-faults", type=positive_int, default=1_000_000)
    parser.add_argument("--max-num-faults", type=positive_int, default=16)
    parser.add_argument("--batch-size", type=positive_int, default=50_000)
    cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    parser.add_argument("--workers", type=positive_int, default=cpu_count)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        collect(args)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; completed batches remain in the checkpoint.\n")


if __name__ == "__main__":
    main()
