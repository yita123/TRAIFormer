"""
RelationFeatureAttention: 分区关系感知变量轴注意力，堆叠 Lf 层
四类关系: pp=0, pq=1, qp=2, qq=3

高效实现：对每种关系类型用 R_mask 选择对应位置，避免大张量 gather。
scores[bt,nh,d,k] = sum_r R_mask_r[d,k] * (Q @ K_r^T / sqrt(h))
H_tilde = sum_r (A * R_mask_r) @ V_r

结构与 model/temporal.py 对称：
  RelationFeatureAttentionLayer  单层计算（原实现，逻辑逐行未变）
  RelationFeatureAttention       Lf 层堆叠容器
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

P_IDX = [0, 1, 2, 3, 4]
Q_IDX = [5, 6, 7, 8, 9, 10]


def build_relation_matrix(D: int, P_idx: list, Q_idx: list) -> torch.Tensor:
    """构建关系矩阵 R [D, D]，R[d,k] ∈ {0(pp),1(pq),2(qp),3(qq)}"""
    R = torch.zeros(D, D, dtype=torch.long)
    p_set = set(P_idx)
    q_set = set(Q_idx)
    for d in range(D):
        for k in range(D):
            if d in p_set and k in p_set:
                R[d, k] = 0
            elif d in p_set and k in q_set:
                R[d, k] = 1
            elif d in q_set and k in p_set:
                R[d, k] = 2
            else:
                R[d, k] = 3
    return R


class RelationFeatureAttentionLayer(nn.Module):
    """单层关系感知变量轴注意力（原 RelationFeatureAttention 实现，逻辑未改动）。"""

    def __init__(self, D: int, H: int, n_heads: int = 4,
                 C: float = 1e4, eps: float = 1e-8,
                 P_idx: list = P_IDX, Q_idx: list = Q_IDX):
        super().__init__()
        assert H % n_heads == 0
        self.D = D
        self.H = H
        self.n_heads = n_heads
        self.h = H // n_heads
        self.C = C
        self.eps = eps
        self.n_relations = 4

        self.W_Q = nn.Linear(H, H, bias=False)
        self.W_K = nn.ModuleList([nn.Linear(H, H, bias=False) for _ in range(4)])
        self.W_V = nn.ModuleList([nn.Linear(H, H, bias=False) for _ in range(4)])
        self.W_O = nn.Linear(H, H, bias=False)

        # 关系类型偏置: 4 个标量
        self.b_r = nn.Parameter(torch.zeros(4))

        # mask modulation
        self.gamma_f = nn.Parameter(torch.zeros(1))

        # Pre-LN
        self.ln1 = nn.LayerNorm(H)
        self.ln2 = nn.LayerNorm(H)

        self.ffn = nn.Sequential(
            nn.Linear(H, 4 * H),
            nn.GELU(),
            nn.Linear(4 * H, H),
        )

        # 对角掩码 [D, D]
        diag_mask = -C * torch.eye(D)
        self.register_buffer("diag_mask", diag_mask)

        # 关系矩阵 [D, D]
        R = build_relation_matrix(D, P_idx, Q_idx)
        self.register_buffer("R", R)

        # 四种关系的 mask [4, D, D]（float，用于 score 加权求和）
        R_masks = torch.stack([(R == r).float() for r in range(4)], dim=0)
        self.register_buffer("R_masks", R_masks)

        # 关系偏置矩阵 [D, D] 用 buffer 缓存（依赖 b_r，需在 forward 中动态计算）
        self.register_buffer("R_long", R)

    def forward(self, H_temp: torch.Tensor, M_in: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H_temp: [B, T, D, H]
            M_in:   [B, T, D]
        Returns:
            H_feat: [B, T, D, H]
        """
        B, T, D, H = H_temp.shape
        n_heads, h = self.n_heads, self.h
        BT = B * T

        # [B, T, D, H] -> [BT, D, H]
        Ht = H_temp.reshape(BT, D, H)
        M_t = M_in.reshape(BT, D)

        Ht_norm = self.ln1(Ht)

        # Q: [BT, D, nh, h] -> [BT, nh, D, h]
        Q = self.W_Q(Ht_norm).view(BT, D, n_heads, h).permute(0, 2, 1, 3)

        # 四套 K/V: 各 [BT, nh, D, h]
        K_list = [self.W_K[r](Ht_norm).view(BT, D, n_heads, h).permute(0, 2, 1, 3)
                  for r in range(4)]
        V_list = [self.W_V[r](Ht_norm).view(BT, D, n_heads, h).permute(0, 2, 1, 3)
                  for r in range(4)]

        # 按关系类型合并 scores: sum_r R_mask_r * (Q @ K_r^T / sqrt(h))
        # 每次 matmul [BT, nh, D, D]，避免大张量展开
        scores = torch.zeros(BT, n_heads, D, D,
                             device=H_temp.device, dtype=H_temp.dtype)
        for r in range(4):
            scores_r = torch.matmul(Q, K_list[r].transpose(-2, -1)) / math.sqrt(h)
            # R_masks[r]: [D, D] -> [1, 1, D, D]
            scores = scores + scores_r * self.R_masks[r].view(1, 1, D, D)

        # 关系偏置: b_r[R[d,k]] -> [D, D] -> [1, 1, D, D]
        b_r_matrix = self.b_r[self.R_long].view(1, 1, D, D)
        scores = scores + b_r_matrix

        # mask modulation
        log_m = torch.log(M_t + self.eps).view(BT, 1, 1, D)
        scores = scores + self.gamma_f * log_m

        # 对角掩码
        scores = scores + self.diag_mask.view(1, 1, D, D)

        A = F.softmax(scores, dim=-1)  # [BT, nh, D, D]

        # H_tilde = sum_r (A * R_mask_r) @ V_r
        H_tilde = torch.zeros(BT, n_heads, D, h,
                              device=H_temp.device, dtype=H_temp.dtype)
        for r in range(4):
            A_r = A * self.R_masks[r].view(1, 1, D, D)
            H_tilde = H_tilde + torch.matmul(A_r, V_list[r])

        # 合并多头: [BT, nh, D, h] -> [BT, D, H]
        H_tilde = H_tilde.permute(0, 2, 1, 3).reshape(BT, D, H)
        H_tilde = self.W_O(H_tilde)
        H_tilde = H_tilde.view(B, T, D, H)

        # 残差来源是 H_temp
        U = H_temp + H_tilde
        H_feat = U + self.ffn(self.ln2(U))
        return H_feat


class RelationFeatureAttention(nn.Module):
    """Lf 层关系感知变量轴注意力堆叠（与 TemporalAttention 的组织方式一致）。

    Lf=1 时行为与重构前的单层实现完全相同（参数名前缀变为 layers.0.，见报告说明）。
    """

    def __init__(self, D: int, H: int, n_heads: int = 4,
                 C: float = 1e4, eps: float = 1e-8,
                 P_idx: list = P_IDX, Q_idx: list = Q_IDX,
                 Lf: int = 1):
        super().__init__()
        if Lf < 1:
            raise ValueError(f"Lf 必须 >= 1，得到 {Lf}")
        self.Lf = Lf
        self.layers = nn.ModuleList([
            RelationFeatureAttentionLayer(D, H, n_heads, C, eps, P_idx, Q_idx)
            for _ in range(Lf)
        ])

    def forward(self, H_temp: torch.Tensor, M_in: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H_temp: [B, T, D, H]
            M_in:   [B, T, D]
        Returns:
            H_feat: [B, T, D, H]
        """
        Z = H_temp
        for layer in self.layers:
            Z = layer(Z, M_in)
        return Z
