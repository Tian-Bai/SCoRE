# SCoRE

[![PyPI package](https://img.shields.io/pypi/v/score-select?label=pypi%20package)](https://pypi.org/project/score-select/)

SCoRE implements conformal selective prediction procedures for marginal
deployment risk (MDR) and selective deployment risk (SDR) control.

This repository also contains the simulation and application code used for the
paper [Conformal Selective Prediction with General Risk Control](https://arxiv.org/abs/2603.24704).

## Installation

```bash
python -m pip install score-select
```

Install the package from a local checkout:

```bash
python -m pip install -e .
```

Install optional dependencies for the research scripts:

```bash
python -m pip install -e ".[experiments]"
```

## Quickstart

```python
import numpy as np
from SCoRE import SCoRE_MDR, SCoRE_SDR

lcalib = np.array([0, 1, 0, 1])
scalib = np.array([0.1, 0.4, 0.2, 0.8])
stest = np.array([0.15, 0.5, 0.9])

dcalib = (lcalib, scalib)
dtest = stest

mdr_selected = SCoRE_MDR(dcalib, dtest, alpha=0.5, gamma=0.5)
sdr_selected = SCoRE_SDR(dcalib, dtest, alpha=0.5, gamma=0.5)
```

Functions return NumPy integer index arrays, so selections can be used directly
to index NumPy arrays.

When using randomized pruning, pass `random_state` for reproducible results:

```python
selected = SCoRE_SDR(
    dcalib,
    dtest,
    alpha=0.5,
    gamma=1.0,
    prune="hete",
    random_state=123,
)
```

## Public API

The top-level package exports the main procedures and utilities:

Recommended package entry points:

- `SCoRE_MDR`
- `SCoRE_SDR`

Additional utilities:

- `CS`
- `SCoRE_MDR_bf`, `SCoRE_MDR_w`, `SCoRE_SDR_w`
- `BH`, `eBH`
- `eval_MDR`, `eval_SDR`
- `loss_Jin2023`, `loss_1`, `loss_2`
- `gen_data_Jin2023`, `gen_data_1`, `gen_data_2`
- `Lpredictor`

## Repository Layout

- `SCoRE/`: installable Python package
- `tests/`: package tests
- `applications/`: real-data applications
  - `applications/drug/`: efficient, cost-aware drug discovery
  - `applications/icu/`: clinical prediction error management
  - `applications/llm/`: flexible LLM abstention
- `simulation/`: simulation experiments
- `simulation_w/`: simulation experiments with covariate shifts
