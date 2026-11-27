"""
Metric functions extracted verbatim from the official SAITS repository
(modeling/utils.py).  Formulas are unchanged.

Source : https://github.com/WenjieDu/SAITS
Commit : 660b87f19c1277065f314f24134f646229e89ca9
License: MIT (see LICENSE_SAITS)

If you use code in this repository, please cite the SAITS paper:

@article{DU2023SAITS,
title = {{SAITS: Self-Attention-based Imputation for Time Series}},
journal = {Expert Systems with Applications},
volume = {219},
pages = {119619},
year = {2023},
doi = {https://doi.org/10.1016/j.eswa.2023.119619},
author = {Wenjie Du and David Cote and Yan Liu},
}
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: MIT

import torch


def masked_mae_cal(inputs, target, mask):
    """calculate Mean Absolute Error"""
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(mask) + 1e-9)


def masked_mse_cal(inputs, target, mask):
    """calculate Mean Square Error"""
    return torch.sum(torch.square(inputs - target) * mask) / (torch.sum(mask) + 1e-9)


def masked_rmse_cal(inputs, target, mask):
    """calculate Root Mean Square Error"""
    return torch.sqrt(masked_mse_cal(inputs, target, mask))


def masked_mre_cal(inputs, target, mask):
    """calculate Mean Relative Error"""
    return torch.sum(torch.abs(inputs - target) * mask) / (
        torch.sum(torch.abs(target * mask)) + 1e-9
    )
