# TRAIFormer

TRAIFormer fills gaps in multivariate industrial time series by combining temporal attention with attention across process and quality variables. This repository contains five PyTorch model implementations and the preprocessed Sulfur Recovery Unit (SRU) dataset.

## Models

Pass one of these names to `models.build_sru_model` to create a model with its SRU settings.

| Model | How it estimates missing values |
| --- | --- |
| M-RNN | Runs recurrent networks forward and backward through the sequence, then combines their estimates with information from other variables. |
| GP-VAE | Uses a variational autoencoder with a Gaussian process prior over latent trajectories. |
| Transformer | Uses a self-attention encoder to estimate missing values from the observed sequence. |
| SAITS | Combines estimates from two diagonally masked self-attention blocks. |
| TRAIFormer | Applies temporal attention within each variable, then relation-aware attention across variables. A learned gate combines the two representations. |

## SRU data

The dataset has 10,081 time points and seven variables. The first five columns are process variables (P), and the last two are quality variables (Q). The series was split chronologically in a 60/20/20 ratio. Windows of 60 steps were then extracted within each split using a stride of 30. Each window belongs to a single split.

| File | Split | Time points | Windows |
| --- | --- | ---: | ---: |
| `data/SRU/train.npz` | First 60% | 6,048 | 200 |
| `data/SRU/val.npz` | Next 20% | 2,016 | 66 |
| `data/SRU/test.npz` | Final 20% | 2,017 | 66 |

Each split stores standardized values in `X` and the original observation mask in `M`, both shaped `(windows, 60, 7)`. A mask value of `1` means observed. `data/SRU/stats.npz` contains the per-variable `mean` and `std` for converting a standardized estimate back to its original units.

## Quick start

From the repository root, install the dependencies in a Python 3.11 environment.

```sh
python -m pip install -r requirements.txt
```

This example hides three observations in the first process variable and fills them with TRAIFormer. The constructor initializes new weights. Use trained weights to obtain meaningful estimates.

```python
import numpy as np
import torch
from models import build_sru_model

with np.load("data/SRU/test.npz") as data:
    x = torch.from_numpy(data["X"][:1])
    visible = torch.from_numpy(data["M"][:1]).bool()

visible[:, 20:23, 0] = False
x_in = torch.where(visible, x, 0.0)

# Time since each variable was last observed, capped at 60 and scaled to [0, 1].
gap = torch.full_like(x[:, 0], 60.0)
delta = torch.empty_like(x)
for t in range(x.shape[1]):
    gap = torch.where(visible[:, t], 0.0, (gap + 1).clamp(max=60))
    delta[:, t] = gap / 60

model = build_sru_model("TRAIFormer").eval()
with torch.inference_mode():
    mean = model(x_in, visible.float(), delta)["mu"]
    filled = torch.where(visible, x, mean)

print(filled.shape)  # torch.Size([1, 60, 7])
```

The baselines have their own input formats. Transformer and SAITS accept a dictionary with `X` and `missing_mask` in test mode. M-RNN uses forward and backward sequences with time gaps. GP-VAE exposes `model.impute(x_in, visible)`. Their implementations and third-party license notices are under `models/`.
