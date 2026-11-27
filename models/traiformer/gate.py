"""
ScalarGate: 轻量标量门控融合时间分支和变量分支
"""

import torch
import torch.nn as nn


class ScalarGate(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入4个标量：H_temp均值, H_feat均值, M_in, delta
        self.gate_linear = nn.Linear(4, 1)

    def forward(self, H_temp: torch.Tensor, H_feat: torch.Tensor,
                M_in: torch.Tensor, delta: torch.Tensor):
        """
        Args:
            H_temp: [B, T, D, H]
            H_feat: [B, T, D, H]
            M_in:   [B, T, D]
            delta:  [B, T, D]
        Returns:
            H_fused: [B, T, D, H]
            g:       [B, T, D]
        """
        h_t_mean = H_temp.mean(-1)  # [B, T, D]
        h_f_mean = H_feat.mean(-1)  # [B, T, D]

        # [B, T, D, 4]
        u = torch.stack([h_t_mean, h_f_mean, M_in, delta], dim=-1)
        g = torch.sigmoid(self.gate_linear(u).squeeze(-1))  # [B, T, D]

        H_fused = g.unsqueeze(-1) * H_temp + (1 - g.unsqueeze(-1)) * H_feat
        return H_fused, g
