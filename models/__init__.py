"""Model constructors for the seven-variable SRU data."""
from .gpvae import BackboneGPVAE
from .mrnn.mit_qbalanced import MRNNMITQBalanced
from .saits.qbalanced import SAITSQBalanced
from .traiformer import TRAIFormer
from .transformer.qbalanced import TransformerQBalanced


def build_sru_model(name, device="cpu"):
    """Construct M-RNN, GP-VAE, Transformer, SAITS, or TRAIFormer for SRU."""
    p, q = list(range(5)), [5, 6]
    if name == "TRAIFormer":
        model = TRAIFormer(T=60, D=7, H=32, n_heads=2, Lt=2, Lf=1, P_idx=p, Q_idx=q)
    elif name == "GP-VAE":
        model = BackboneGPVAE(input_dim=7, time_length=60, latent_dim=8,
                              window_size=15, beta=0.2)
        model.prior = model._init_prior(device=device)
    elif name == "M-RNN":
        model = MRNNMITQBalanced(seq_len=60, feature_num=7, rnn_hidden_size=64,
                                 p_idx=p, q_idx=q, q_weight=2.5, MIT=True, device=device)
    elif name in ("SAITS", "Transformer"):
        cls = SAITSQBalanced if name == "SAITS" else TransformerQBalanced
        model = cls(n_groups=1 if name == "SAITS" else 2, n_group_inner_layers=1,
                    d_time=60, d_feature=7, d_model=32, d_inner=128, n_head=2,
                    d_k=16, d_v=16, dropout=0.1, input_with_mask=True,
                    param_sharing_strategy="inner_group", diagonal_attention_mask=True,
                    p_idx=p, q_idx=q, q_weight=2.5, MIT=True, device=device)
    else:
        raise ValueError(f"Unknown SRU model: {name}")
    return model.to(device)
