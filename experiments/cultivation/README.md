# Magic State Cultivation

These scripts reproduce Figure 13 from Gidney, Shutty, and Jones' "Magic state cultivation:
growing T states as cheap as CNOT gates" ([arXiv:2409.17595](https://arxiv.org/abs/2409.17595)),
using importance sampling as described [here](https://unitaryfoundation.github.io/clifft/stable/guide/importance-sampling-tutorial/).

From the repository root, install the experiment dependencies and build the simulator:

```sh
uv sync --group experiments
```

Collect samples for each circuit independently, choosing the number of worker processes.

```sh
uv run --group experiments python experiments/cultivation/sample.py --circuit t --workers 8
uv run --group experiments python experiments/cultivation/sample.py --circuit s --workers 8
```

The collector writes `samples_t.json` or `samples_s.json` alongside the scripts.
Use `--output PATH` for separate experiments. Repeating a command resumes its
checkpoint; increasing `--shots-per-num-faults` adds samples. 

To collect the same experiment using Clifft:

```sh
uv run --group experiments python experiments/cultivation/sample.py --circuit t --backend clifft --workers 8
uv run --group experiments python experiments/cultivation/sample.py --circuit s --backend clifft --workers 8
```

These commands write `samples_clifft_t.json` and `samples_clifft_s.json`.
Afterwards, generate the plot by running:

```sh
uv run --group experiments python experiments/cultivation/plot.py
```

The outputs are `cultivation.png` and `cultivation.pdf`. Override
`--output-prefix` or `--physical-error-rates 0.001 0.003 0.01` as needed.
