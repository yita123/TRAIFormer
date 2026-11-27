"""P/Q-weighted reconstruction and imputation losses."""

import torch

from .model import SAITS
from .utils import masked_mae_cal
from ..pq_weighting import register_element_weights, weighted_mae


def masked_mae_pq_balanced(pred, target, mask, p_idx, q_idx):
    """按 P / Q 两组分别求 MAE 后等权平均。

    * 两组都有有效位置 → (L_p + L_q) / 2
    * 只有一组有有效位置 → 只用该组
    * 两组都没有有效位置 → 返回 0（与 masked_mae_cal 在空掩码下的行为一致，不产生 NaN）
    """
    m_p, m_q = mask[..., p_idx], mask[..., q_idx]
    n_p, n_q = m_p.sum(), m_q.sum()

    has_p, has_q = bool(n_p > 0), bool(n_q > 0)
    if has_p and has_q:
        l_p = masked_mae_cal(pred[..., p_idx], target[..., p_idx], m_p)
        l_q = masked_mae_cal(pred[..., q_idx], target[..., q_idx], m_q)
        return (l_p + l_q) / 2.0
    if has_p:
        return masked_mae_cal(pred[..., p_idx], target[..., p_idx], m_p)
    if has_q:
        return masked_mae_cal(pred[..., q_idx], target[..., q_idx], m_q)
    # 两组均无有效位置：返回可反向传播的 0，避免 0/0 产生 NaN
    return (pred.sum() * 0.0)


class SAITSQBalanced(SAITS):
    """结构与 SAITS 完全相同，只改损失聚合（P / Q 等权）。"""

    def __init__(self, *args, p_idx=None, q_idx=None, q_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        if p_idx is None or q_idx is None:
            raise ValueError("SAITSQBalanced 需要 p_idx 与 q_idx")
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
        """与官方 forward 同构，仅把 masked_mae_cal 换成 P/Q 平衡版本。"""
        X, masks = inputs["X"], inputs["missing_mask"]
        imputed_data, [X_tilde_1, X_tilde_2, X_tilde_3] = self.impute(inputs)

        reconstruction_loss = 0
        reconstruction_loss += self._bal(X_tilde_1, X, masks)
        reconstruction_loss += self._bal(X_tilde_2, X, masks)
        final_reconstruction_MAE = self._bal(X_tilde_3, X, masks)
        reconstruction_loss += final_reconstruction_MAE
        reconstruction_loss /= 3

        if (self.MIT or stage == "val") and stage != "test":
            imputation_MAE = self._bal(X_tilde_3, inputs["X_holdout"],
                                       inputs["indicating_mask"])
        else:
            imputation_MAE = torch.tensor(0.0)

        return {
            "imputed_data": imputed_data,
            "reconstruction_loss": reconstruction_loss,
            "imputation_loss": imputation_MAE,
            "reconstruction_MAE": final_reconstruction_MAE,
            "imputation_MAE": imputation_MAE,
        }
