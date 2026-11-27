"""P/Q-weighted M-RNN reconstruction and imputation losses."""

import torch

from .mit import MRNNMIT
from .utils import masked_mae_cal, masked_rmse_cal
from ..pq_weighting import register_element_weights, weighted_mae, weighted_rmse


class MRNNMITQBalanced(MRNNMIT):
    """结构与 MRNNMIT 完全相同，只改 ORT/MIT 的 P/Q 损失聚合。"""

    def __init__(self, seq_len, feature_num, rnn_hidden_size,
                 p_idx=None, q_idx=None, q_weight=None, **kwargs):
        # 先按 MRNNMIT（= 官方 MRNN）构造全部子模块，
        # 构造顺序与随机数消耗完全一致 → 同种子下初始权重逐元素相同
        super().__init__(seq_len, feature_num, rnn_hidden_size, **kwargs)
        if p_idx is None or q_idx is None:
            raise ValueError("MRNNMITQBalanced 需要 p_idx 与 q_idx（从配置读取）")
        p_idx, q_idx = list(p_idx), list(q_idx)
        overlap = sorted(set(p_idx) & set(q_idx))
        if overlap:
            raise ValueError(f"p_idx 与 q_idx 不得重叠，重叠列: {overlap}")
        bad = [i for i in p_idx + q_idx if not (0 <= i < feature_num)]
        if bad:
            raise ValueError(f"变量下标越界（feature_num={feature_num}）: {bad}")
        col_p = torch.zeros(feature_num, dtype=torch.float32)
        col_q = torch.zeros(feature_num, dtype=torch.float32)
        col_p[p_idx] = 1.0
        col_q[q_idx] = 1.0
        # persistent=False：不进 state_dict，checkpoint 结构与 MRNNMIT 完全一致
        self.register_buffer("col_p", col_p, persistent=False)
        self.register_buffer("col_q", col_q, persistent=False)
        self.p_idx, self.q_idx = p_idx, q_idx
        register_element_weights(self, feature_num, q_idx, q_weight)

    # ── P/Q 等权聚合 ─────────────────────────────────────────────────────────

    def _balanced(self, fn, pred, target, mask):
        """用官方 fn（masked_rmse_cal / masked_mae_cal）按 P/Q 分组等权聚合。

        `mask` 形状可为 [B, D]（逐时刻 ORT）或 [B, T, D]（MIT），
        列掩码 [D] 依广播规则作用在最后一维。
        """
        m_p = mask * self.col_p
        m_q = mask * self.col_q
        has_p = bool(m_p.sum() > 0)
        has_q = bool(m_q.sum() > 0)
        if has_p and has_q:
            return (fn(pred, target, m_p) + fn(pred, target, m_q)) / 2.0
        if has_p:
            return fn(pred, target, m_p)
        if has_q:
            return fn(pred, target, m_q)
        # 两组均无有效位置：返回可反向传播的有限 0，避免 0/0 产生 NaN
        return pred.sum() * 0.0

    def masked_rmse_pq_balanced(self, pred, target, mask):
        if self.pq_element_weights is not None:
            return weighted_rmse(pred, target, mask, self.pq_element_weights)
        return self._balanced(masked_rmse_cal, pred, target, mask)

    def masked_mae_pq_balanced(self, pred, target, mask):
        if self.pq_element_weights is not None:
            return weighted_mae(pred, target, mask, self.pq_element_weights)
        return self._balanced(masked_mae_cal, pred, target, mask)

    # ── ORT ──────────────────────────────────────────────────────────────────

    def impute(self, data):
        """官方 `MRNN.impute` 的逐行复制，**仅 ORT 一行的聚合方式不同**。

        两路估计（RNN_estimation / FCN_estimation）、逐时刻累计顺序、
        `estimations` 与 `imputed_data` 的构造全部与官方一致；
        `forward()` 中的 `/= seq_len` 也不变。ORT 仍然使用 RMSE。
        """
        hidden_states_f = self.gene_hidden_states(data, "forward")
        hidden_states_b = self.gene_hidden_states(data, "backward")[::-1]

        values = data["forward"]["X"]
        masks = data["forward"]["missing_mask"]

        reconstruction_loss = 0
        estimations = []
        for i in range(
            self.seq_len
        ):  # calculating estimation loss for times can obtain better results than once
            x = values[:, i, :]
            m = masks[:, i, :]
            h_f = hidden_states_f[i]
            h_b = hidden_states_b[i]
            h = torch.cat([h_f, h_b], dim=1)
            RNN_estimation = self.concated_hidden_project(h)  # x̃_t
            RNN_imputed_data = m * x + (1 - m) * RNN_estimation
            FCN_estimation = self.fcn_regression(
                x, m, RNN_imputed_data
            )  # FCN estimation is output extimation
            reconstruction_loss += self.masked_rmse_pq_balanced(   # ← 官方: masked_rmse_cal
                FCN_estimation, x, m
            ) + self.masked_rmse_pq_balanced(                      # ← 官方: masked_rmse_cal
                RNN_estimation, x, m
            )
            estimations.append(FCN_estimation.unsqueeze(dim=1))

        estimations = torch.cat(estimations, dim=1)
        imputed_data = masks * values + (1 - masks) * estimations
        return imputed_data, [estimations, reconstruction_loss]

    # ── MIT ──────────────────────────────────────────────────────────────────

    def forward(self, data, stage):
        """`mit.py::MRNNMIT.forward` 的逐行复制，**仅两处 MAE 聚合方式不同**。

        MIT 启用条件与 MRNNMIT 相同；`reconstruction_MAE` 仍带 `.detach()`。
        """
        values = data["forward"]["X"]
        masks = data["forward"]["missing_mask"]
        imputed_data, [estimations, reconstruction_loss] = self.impute(data)
        reconstruction_loss /= self.seq_len
        # 仅监控指标：保留 .detach()，不产生梯度
        reconstruction_MAE = self.masked_mae_pq_balanced(          # ← 原: masked_mae_cal
            estimations.detach(), values, masks
        )

        if (self.MIT or stage == "val") and stage != "test":
            # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
            imputation_MAE = self.masked_mae_pq_balanced(          # ← 原: masked_mae_cal
                imputed_data, data["X_holdout"], data["indicating_mask"]
            )
        else:
            imputation_MAE = torch.tensor(0.0)

        ret_dict = {
            "reconstruction_loss": reconstruction_loss,
            "reconstruction_MAE": reconstruction_MAE,
            "imputation_loss": imputation_MAE,
            "imputation_MAE": imputation_MAE,
            "imputed_data": imputed_data,
        }
        if "X_holdout" in data:
            ret_dict["X_holdout"] = data["X_holdout"]
            ret_dict["indicating_mask"] = data["indicating_mask"]
        return ret_dict
