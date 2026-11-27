"""
TemporalAttention: 对每个变量 d 独立在时间维度做 self-attention，堆叠 Lt 层
手写注意力以保证 MPS 兼容性
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionLayer(nn.Module):
    def __init__(self, T: int, D: int, H: int, n_heads: int, C: float = 1e4,
                 eps: float = 1e-8):
        super().__init__()
        assert H % n_heads == 0
        self.T = T
        self.D = D
        self.H = H
        self.n_heads = n_heads
        self.h = H // n_heads
        self.C = C
        self.eps = eps

        self.W_Q = nn.Linear(H, H, bias=False)
        self.W_K = nn.Linear(H, H, bias=False)
        self.W_V = nn.Linear(H, H, bias=False)
        self.W_O = nn.Linear(H, H, bias=False)

        # 时间滞后偏置: Embedding(2T-1, n_heads)
        self.lag_bias = nn.Embedding(2 * T - 1, n_heads)

        # mask modulation 权重（每头一个，共 n_heads 个标量）
        self.gamma_t = nn.Parameter(torch.zeros(n_heads))

        # Pre-LN
        self.ln1 = nn.LayerNorm(H)
        self.ln2 = nn.LayerNorm(H)

        # FFN: H -> 4H -> H
        self.ffn = nn.Sequential(
            nn.Linear(H, 4 * H),
            nn.GELU(),
            nn.Linear(4 * H, H),
        )

        # 对角掩码（固定）
        diag_mask = -C * torch.eye(T)  # [T, T]
        self.register_buffer("diag_mask", diag_mask)

        # 滞后索引（固定）
        t_idx = torch.arange(T)
        lag_idx = t_idx.unsqueeze(1) - t_idx.unsqueeze(0) + (T - 1)  # [T, T]
        self.register_buffer("lag_idx", lag_idx)

    def forward(self, Z: torch.Tensor, M_in: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z:    [B, T, D, H]
            M_in: [B, T, D]
        Returns:
            Z_out: [B, T, D, H]
        """
        B, T, D, H = Z.shape
        n_heads, h = self.n_heads, self.h

        # Pre-LN on the attention branch
        Z_norm = self.ln1(Z)

        # Reshape to [B*D, T, H] for temporal attention
        Z_bd = Z_norm.permute(0, 2, 1, 3).reshape(B * D, T, H)

        Q = self.W_Q(Z_bd).view(B * D, T, n_heads, h).transpose(1, 2)  # [B*D, nh, T, h]
        K = self.W_K(Z_bd).view(B * D, T, n_heads, h).transpose(1, 2)
        V = self.W_V(Z_bd).view(B * D, T, n_heads, h).transpose(1, 2)

        # 注意力得分: [B*D, nh, T, T]
        scale = math.sqrt(h)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # 时间滞后偏置: lag_bias[lag_idx] -> [T, T, nh] -> [1, nh, T, T]
        b_lag = self.lag_bias(self.lag_idx)  # [T, T, nh]
        b_lag = b_lag.permute(2, 0, 1).unsqueeze(0)  # [1, nh, T, T]
        scores = scores + b_lag

        # mask modulation: gamma_t * log(M_in[b, τ, d] + eps)
        # M_in: [B, T, D] -> [B, D, T] -> [B*D, T] -> [B*D, 1, 1, T]
        M_bd = M_in.permute(0, 2, 1).reshape(B * D, T)
        log_m = torch.log(M_bd + self.eps)  # [B*D, T]
        # gamma_t: [nh] -> [1, nh, 1, 1]
        gamma = self.gamma_t.view(1, n_heads, 1, 1)
        # log_m: [B*D, T] -> [B*D, 1, 1, T]
        log_m = log_m.unsqueeze(1).unsqueeze(2)
        scores = scores + gamma * log_m

        # 对角掩码: [1, 1, T, T]
        scores = scores + self.diag_mask.view(1, 1, T, T)

        A = F.softmax(scores, dim=-1)  # [B*D, nh, T, T]

        # 加权聚合
        H_attn = torch.matmul(A, V)  # [B*D, nh, T, h]
        H_attn = H_attn.transpose(1, 2).reshape(B * D, T, H)
        H_attn = self.W_O(H_attn)

        # 还原为 [B, T, D, H]
        H_attn = H_attn.view(B, D, T, H).permute(0, 2, 1, 3)

        # 残差 1（Pre-LN 已在前）
        U = Z + H_attn

        # FFN + 残差 2
        Z_out = U + self.ffn(self.ln2(U))
        return Z_out


class TemporalAttention(nn.Module):
    def __init__(self, T: int, D: int, H: int, n_heads: int = 4,
                 Lt: int = 4, C: float = 1e4, eps: float = 1e-8):
        super().__init__()
        if Lt < 1:
            raise ValueError(f"Lt 必须 >= 1，得到 {Lt}")
        self.Lt = Lt
        self.layers = nn.ModuleList([
            TemporalAttentionLayer(T, D, H, n_heads, C, eps)
            for _ in range(Lt)
        ])

    def forward(self, Z0: torch.Tensor, M_in: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z0:    [B, T, D, H]
            M_in:  [B, T, D]
        Returns:
            H_temp: [B, T, D, H]
        """
        Z = Z0
        for layer in self.layers:
            Z = layer(Z, M_in)
        return Z
