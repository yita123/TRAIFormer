"""
URAIT: Unified Relation-aware Axial Imputation Transformer
整体模型，组装所有子模块
"""

import torch
import torch.nn as nn
from .embedding import InputEmbedding
from .temporal import TemporalAttention
from .feature import RelationFeatureAttention
from .gate import ScalarGate
from .head import OutputHead

P_IDX = [0, 1, 2, 3, 4]
Q_IDX = [5, 6, 7, 8, 9, 10]


class URAIT(nn.Module):
    """
    数据流（不随 Lt / Lf 变化）：
        InputEmbedding
          → TemporalAttention            × Lt   （时间轴）
          → RelationFeatureAttention     × Lf   （关系感知变量轴）
          → ScalarGate                          （标量门控融合）
          → OutputHead                          （均值 / 方差头）

    Args:
        Lt: 时间轴注意力堆叠层数
        Lf: 关系感知变量轴注意力堆叠层数
    """

    def __init__(self, T: int = 48, D: int = 11, H: int = 128,
                 n_heads: int = 4, Lt: int = 4, Lf: int = 1,
                 C: float = 1e4,
                 sigma_min: float = 0.01, eps: float = 1e-8,
                 P_idx: list = P_IDX, Q_idx: list = Q_IDX):
        super().__init__()
        # 统一校验：必须在创建任何子模块之前，避免构造出空层列表
        if Lt < 1:
            raise ValueError(f"Lt 必须 >= 1，得到 {Lt}")
        if Lf < 1:
            raise ValueError(f"Lf 必须 >= 1，得到 {Lf}")
        self.Lt = Lt
        self.Lf = Lf
        self.embedding = InputEmbedding(T=T, D=D, H=H, P_idx=P_idx, Q_idx=Q_idx)
        self.temporal = TemporalAttention(T=T, D=D, H=H, n_heads=n_heads,
                                          Lt=Lt, C=C, eps=eps)
        self.feature = RelationFeatureAttention(D=D, H=H, n_heads=n_heads,
                                                C=C, eps=eps,
                                                P_idx=P_idx, Q_idx=Q_idx,
                                                Lf=Lf)
        self.gate = ScalarGate()
        self.head = OutputHead(H=H, sigma_min=sigma_min)

    def forward(self, X_in: torch.Tensor, M_in: torch.Tensor,
                delta: torch.Tensor) -> dict:
        """
        Args:
            X_in:  [B, T, D]
            M_in:  [B, T, D]
            delta: [B, T, D]
        Returns:
            dict with keys: mu, sigma2, conf — each [B, T, D]
        """
        Z0 = self.embedding(X_in, M_in, delta)
        H_temp = self.temporal(Z0, M_in)
        H_feat = self.feature(H_temp, M_in)
        H_fused, g = self.gate(H_temp, H_feat, M_in, delta)
        mu, sigma2, conf = self.head(H_fused)
        return {"mu": mu, "sigma2": sigma2, "conf": conf}
