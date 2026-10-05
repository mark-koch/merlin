# Benchmarks

## Setup

For benchmarking against SymFt, we include a small patch to allow SymFT to
build on Apple Silicon. In the root of the repository, run

```console
git submodule update --init benchmarks/symft
git -C benchmarks/symft apply ../symft-arm64.patch
```

Then, install depenencies using

```console
uv sync --group benchmark
```

## Bravyi-Haah 3k-to-1 Distillation

Delete the existing results checkpoint:

```console
rm benchmarks/results/bh.json
```

Then, run

```console
uv run python -m benchmarks run --cases bh:2,bh:4,bh:6,bh:8,bh:10,bh:12,bh:14,bh:16,bh:18 --p-grid 0.001 --workload detector-sampling --sim merlin,clifft,symft,xtim --shots 1000 --out benchmarks/results/bh.json
```

Since tsim runs out of memory quickly for larger circuit instances, we only run it for smaller circuits:

```console
uv run python -m benchmarks run --cases bh:2,bh:4,bh:6,bh:8 --p-grid 0.001 --workload detector-sampling --sim tsim --shots 1000 --out benchmarks/results/bh.json
```

To plot the results, run

```
uv run python -m benchmarks plot benchmarks/results/bh.json --metric total-time --x k --log-y --workload detector-sampling --save benchmarks/results/bh-total-time.pdf --no-show
uv run python -m benchmarks plot benchmarks/results/bh.json --metric sample-time-per-attempt --x k --log-y --workload detector-sampling --save benchmarks/results/bh-sample-time.pdf --no-show
uv run python -m benchmarks plot benchmarks/results/bh.json --metric peak-rss --x k --log-y --workload detector-sampling --save benchmarks/results/bh-memory.pdf --no-show
```


## Other Circuits

Delete the existing results checkpoint:

```console
rm benchmarks/results/other.json
```

Then, run

```console
uv run python -m benchmarks run --cases 15to1 --p-grid 0.001 --workload detector-sampling --sim merlin,clifft,symft,xtim,tsim --shots 1000 --out benchmarks/results/other.json
uv run python -m benchmarks run --cases cultivation-d3-t,cultivation-d5-t --p-grid 0.001 --workload detector-sampling --sim merlin,clifft,symft,xtim --shots 1000 --out benchmarks/results/other.json
uv run python -m benchmarks run --cases bt27 --p-grid 0.001 --workload detector-sampling --sim merlin,xtim --shots 1000 --out benchmarks/results/other.json
uv run python -m benchmarks run --cases bt81 --p-grid 0.001 --workload detector-sampling --sim merlin --shots 1000 --out benchmarks/results/other.json
```

Again, we have exlcluded simulator-circuit combinations that run out of memory in our tests.

To output the results, run

```console
uv run benchmarks/print_results.py
```
