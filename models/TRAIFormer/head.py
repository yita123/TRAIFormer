"""
OutputHead: 从融合表示预测均值、方差、置信度

方差的正值变换（softplus + sigma_min 下界）与 confidence 的换算被提取为两个
模块级函数，供 `OutputHead` 与只出方差的 `VarianceHead` 共用 —— 运算与顺序
逐行未变，`OutputHead` 的数值、参数与构造顺序（mu_head → rho_head）
与提取前完全相同。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sigma2_from_rho(rho: torch.Tensor, sigma_min: float) -> torch.Tensor:
    """方差的正值变换：softplus(rho) + sigma_min（全项目唯一定义）。"""
    return F.softplus(rho) + sigma_min


def conf_from_sigma2(sigma2: torch.Tensor) -> torch.Tensor:
    """confidence = 1 / (1 + sqrt(sigma2))（全项目唯一定义）。"""
    return 1.0 / (1.0 + torch.sqrt(sigma2))


class OutputHead(nn.Module):
    def __init__(self, H: int, sigma_min: float = 0.01):
        super().__init__()
        self.sigma_min = sigma_min
        self.mu_head = nn.Linear(H, 1)
        self.rho_head = nn.Linear(H, 1)

    def forward(self, H_fused: torch.Tensor):
        """
        Args:
            H_fused: [B, T, D, H]
        Returns:
            mu:     [B, T, D]
            sigma2: [B, T, D]
            conf:   [B, T, D]
        """
        mu = self.mu_head(H_fused).squeeze(-1)
        rho = self.rho_head(H_fused).squeeze(-1)
        sigma2 = sigma2_from_rho(rho, self.sigma_min)
        conf = conf_from_sigma2(sigma2)
        return mu, sigma2, conf


class VarianceHead(nn.Module):
    """只出方差 / confidence 的预测头。

    用于均值由**残差修正**路径给出、因而不需要 `OutputHead.mu_head` 的变体
    （见 `model/coarse_patch_refine.py`）—— 这样新模型里不会留下任何
    「不参与 forward 的旧均值头参数」。

    方差的正值变换、下界与 confidence 换算与 `OutputHead` 共用同一实现，
    未对方差做无约束的加法残差。
    """

    def __init__(self, H: int, sigma_min: float = 0.01):
        super().__init__()
        self.sigma_min = sigma_min
        self.rho_head = nn.Linear(H, 1)

    def forward(self, H_in: torch.Tensor):
        """
        Args:
            H_in: [B, T, D, H]
        Returns:
            sigma2: [B, T, D]
            conf:   [B, T, D]
        """
        rho = self.rho_head(H_in).squeeze(-1)
        sigma2 = sigma2_from_rho(rho, self.sigma_min)
        conf = conf_from_sigma2(sigma2)
        return sigma2, conf
