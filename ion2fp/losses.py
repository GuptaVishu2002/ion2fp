import torch.nn.functional as F


def compute_loss(pred, target, loss_type):
    if loss_type in ("bce", "binary_cross_entropy"):
        return F.binary_cross_entropy(pred, target)
    if loss_type == "cosine_similarity":
        return 1 - F.cosine_similarity(pred, target).mean()
    if loss_type == "tanimoto":
        common = (pred * target).sum(dim=1)
        p = (pred * pred).sum(dim=1)
        t = (target * target).sum(dim=1)
        return 1 - (common / (p + t - common).clamp(min=1e-8)).mean()
    raise ValueError(f"unknown loss_type {loss_type!r}, pick bce, cosine_similarity, or tanimoto")
