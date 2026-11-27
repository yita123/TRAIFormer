import torch
from torch import nn

Q_IDX = [5, 6]
EPS = 1e-8


class URATLoss(nn.Module):
    def __init__(self, lambda_q: float = 2.5, beta: float = 0.5,
                 lambda_ort: float = 0.2,
                 Q_idx: list = Q_IDX, D: int = 7,
                 eps: float = EPS):
        super().__init__()
        self.beta = beta
        self.lambda_ort = lambda_ort
        self.eps = eps

        # 变量权重: Q 变量放大 lambda_q 倍
        w = torch.ones(D)
        w[Q_idx] = lambda_q
        self.register_buffer("w", w)

    def forward(self, mu: torch.Tensor, sigma2: torch.Tensor,
                X: torch.Tensor, M: torch.Tensor, I: torch.Tensor) -> dict:
        """
        Args:
            mu:     [B, T, D]  预测均值
            sigma2: [B, T, D]  预测方差
            X:      [B, T, D]  真实值
            M:      [B, T, D]  原始观测掩码
            I:      [B, T, D]  人工遮挡掩码
        Returns:
            dict: loss (总损失), mae_mit, nll_mit, ort
        """
        w = self.w  # [D]

        # MIT-MAE：只在 I=1 位置
        mae = (mu - X).abs()  # [B, T, D]
        w_I = I * w  # [B, T, D]
        denom_mit = w_I.sum() + self.eps
        L_MAE_MIT = (w_I * mae).sum() / denom_mit

        # MIT-NLL：只在 I=1 位置
        nll = (X - mu) ** 2 / (2.0 * sigma2) + 0.5 * torch.log(sigma2)
        L_NLL_MIT = (w_I * nll).sum() / denom_mit

        # ORT：在 M=1, I=0 位置
        ort_mask = M * (1.0 - I)
        w_ort = ort_mask * w
        denom_ort = w_ort.sum() + self.eps
        L_ORT = (w_ort * mae).sum() / denom_ort

        L = L_MAE_MIT + self.beta * L_NLL_MIT + self.lambda_ort * L_ORT

        return {
            "loss": L,
            "mae_mit": L_MAE_MIT,
            "nll_mit": L_NLL_MIT,
            "ort": L_ORT,
        }
