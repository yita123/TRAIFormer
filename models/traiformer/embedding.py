"""
InputEmbedding: 将 X_in, M_in, delta 映射为统一嵌入 Z0 [B, T, D, H]

Z0[b,t,d] = Ex(X_in[b,t,d])       Linear(1->H)
           + Em(M_in[b,t,d])       Embedding(2, H)
           + E_delta(delta[b,t,d]) Linear(1->H)
           + et[t]                 Embedding(T, H)
           + ed[d]                 Embedding(D, H)
           + eg[gd]                Embedding(2, H)  P=0, Q=1
"""

import torch
import torch.nn as nn

P_IDX = [0, 1, 2, 3, 4]
Q_IDX = [5, 6, 7, 8, 9, 10]


class InputEmbedding(nn.Module):
    def __init__(self, T: int, D: int, H: int,
                 P_idx: list = P_IDX, Q_idx: list = Q_IDX):
        super().__init__()
        self.T = T
        self.D = D
        self.H = H

        self.Ex = nn.Linear(1, H)
        self.Em = nn.Embedding(2, H)
        self.E_delta = nn.Linear(1, H)
        self.et = nn.Embedding(T, H)
        self.ed = nn.Embedding(D, H)
        self.eg = nn.Embedding(2, H)

        # 变量类型 ids: [D]，P=0, Q=1
        type_ids = torch.zeros(D, dtype=torch.long)
        for idx in Q_idx:
            type_ids[idx] = 1
        self.register_buffer("type_ids", type_ids)

        # 位置索引（固定）
        t_ids = torch.arange(T)
        d_ids = torch.arange(D)
        self.register_buffer("t_ids", t_ids)
        self.register_buffer("d_ids", d_ids)

    def forward(self, X_in: torch.Tensor, M_in: torch.Tensor,
                delta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X_in:  [B, T, D]
            M_in:  [B, T, D]
            delta: [B, T, D]
        Returns:
            Z0: [B, T, D, H]
        """
        B, T, D = X_in.shape

        # 数值嵌入：[B, T, D, 1] -> [B, T, D, H]
        z_x = self.Ex(X_in.unsqueeze(-1))

        # 掩码嵌入：M_in 转 long -> [B, T, D, H]
        z_m = self.Em(M_in.long())

        # 时间差嵌入：[B, T, D, 1] -> [B, T, D, H]
        z_delta = self.E_delta(delta.unsqueeze(-1))

        # 时间位置编码：[T, H] -> [1, T, 1, H]
        z_t = self.et(self.t_ids).view(1, T, 1, self.H)

        # 变量 ID 编码：[D, H] -> [1, 1, D, H]
        z_d = self.ed(self.d_ids).view(1, 1, D, self.H)

        # 变量类型编码：[D, H] -> [1, 1, D, H]
        z_g = self.eg(self.type_ids).view(1, 1, D, self.H)

        Z0 = z_x + z_m + z_delta + z_t + z_d + z_g
        return Z0
