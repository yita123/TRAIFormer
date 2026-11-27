from .backbone import BackboneGPVAE


def loss(model: BackboneGPVAE, x, visible, weights):
    # Same single-sample negative ELBO as PyPOTS. Only the reconstruction
    # likelihood is feature-weighted. Masked-out values never enter the encoder.
    posterior = model.encode(x)
    decoded = model.decode(posterior.rsample())
    reconstruction = (-decoded.log_prob(x) * visible * weights).sum(dim=(1, 2))
    kl = model.kl_divergence(posterior, model.prior).sum(1)
    return (reconstruction + model.beta * kl).mean()
