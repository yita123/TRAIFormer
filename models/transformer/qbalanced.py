"""P/Q-weighted reconstruction and imputation losses."""

import torch

from .model import TransformerEncoder
from .utils import masked_mae_cal
from ..pq_weighting import register_element_weights, weighted_mae


def masked_mae_pq_balanced(pred, target, mask, p_idx, q_idx):
    """按 P / Q 两组分别求 MAE 后等权平均；空掩码安全，永不返回 NaN/Inf。"""
    m_p, m_q = mask[..., p_idx], mask[..., q_idx]
    has_p, has_q = bool(m_p.sum() > 0), bool(m_q.sum() > 0)

    if has_p and has_q:
        l_p = masked_mae_cal(pred[..., p_idx], target[..., p_idx], m_p)
        l_q = masked_mae_cal(pred[..., q_idx], target[..., q_idx], m_q)
        return (l_p + l_q) / 2.0
    if has_p:
        return masked_mae_cal(pred[..., p_idx], target[..., p_idx], m_p)
    if has_q:
        return masked_mae_cal(pred[..., q_idx], target[..., q_idx], m_q)
    # 两组均无有效位置：返回可反向传播的 0，避免 0/0
    return pred.sum() * 0.0


class TransformerQBalanced(TransformerEncoder):
    """结构与官方 TransformerEncoder 完全相同，只改损失聚合（P / Q 等权）。"""

    def __init__(self, *args, p_idx=None, q_idx=None, q_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        if p_idx is None or q_idx is None:
            raise ValueError("TransformerQBalanced 需要 p_idx 与 q_idx（从配置读取）")
        # 纯 Python 列表，不注册为 buffer/parameter → 参数量保持不变
        self.p_idx = list(p_idx)
        self.q_idx = list(q_idx)
        if q_weight is None:
            self.pq_element_weights = None
        else:
            register_element_weights(self, kwargs["d_feature"], self.q_idx, q_weight)

    def _bal(self, pred, target, mask):
        if self.pq_element_weights is not None:
            return weighted_mae(pred, target, mask, self.pq_element_weights)
        return masked_mae_pq_balanced(pred, target, mask, self.p_idx, self.q_idx)

    def forward(self, inputs, stage):
        """与官方 forward 同构，仅把 masked_mae_cal 换成 P/Q 等权版本。"""
        X, masks = inputs["X"], inputs["missing_mask"]
        imputed_data, learned_presentation = self.impute(inputs)

        reconstruction_MAE = self._bal(learned_presentation, X, masks)
        if (self.MIT or stage == "val") and stage != "test":
            imputation_MAE = self._bal(learned_presentation, inputs["X_holdout"],
                                       inputs["indicating_mask"])
        else:
            imputation_MAE = torch.tensor(0.0)

        return {
            "imputed_data": imputed_data,
            "reconstruction_loss": reconstruction_MAE,
            "imputation_loss": imputation_MAE,
            "reconstruction_MAE": reconstruction_MAE,
            "imputation_MAE": imputation_MAE,
        }
