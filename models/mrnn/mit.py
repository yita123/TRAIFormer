"""Masked imputation training for M-RNN."""

import torch

from .model import MRNN
from .utils import masked_mae_cal


class MRNNMIT(MRNN):
    """结构与官方 MRNN 完全相同，只让训练阶段真正启用 MIT。"""

    def __init__(self, seq_len, feature_num, rnn_hidden_size, **kwargs):
        # 官方 MRNN.__init__ 只读 kwargs["device"]，此处原样调用，
        # 子模块创建顺序与随机数消耗完全一致 → 同种子下初始权重逐元素相同
        super().__init__(seq_len, feature_num, rnn_hidden_size, **kwargs)
        # 官方 MRNN 不设置 self.MIT，这正是训练期 MIT 失效的根源；本类补上
        self.MIT = kwargs.get("MIT", True)

    # gene_hidden_states / impute 全部继承官方实现，未做任何改动。

    def forward(self, data, stage):
        """官方 `MRNN.forward` 的逐行复制，**仅 MIT 判断一行不同**。"""
        values = data["forward"]["X"]
        masks = data["forward"]["missing_mask"]
        imputed_data, [estimations, reconstruction_loss] = self.impute(data)
        reconstruction_loss /= self.seq_len
        reconstruction_MAE = masked_mae_cal(estimations.detach(), values, masks)

        if (self.MIT or stage == "val") and stage != "test":   # ← 官方: if stage == "val":
            # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
            imputation_MAE = masked_mae_cal(
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
