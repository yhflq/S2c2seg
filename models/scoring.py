import torch


EPS_NORM = 1e-6
EPS_LOG = 1e-8


def minmax(scores, eps=EPS_NORM):
    lo = scores.min()
    hi = scores.max()
    return (scores - lo) / (hi - lo + eps)


def global_alignment(dense_logits, mode="patch_mean", cls_scores=None):
    if mode == "cls":
        if cls_scores is None:
            raise ValueError("global_score='cls' requires cls_scores")
        return cls_scores.float()
    if dense_logits.ndim != 3:
        raise ValueError("dense_logits must be [Q, H, W], got %r"
                         % (tuple(dense_logits.shape),))
    return dense_logits.float().mean(dim=(-2, -1))


def spatial_presence(local_probs):
    if local_probs.ndim != 3:
        raise ValueError("local_probs must be [Q, H, W], got %r"
                         % (tuple(local_probs.shape),))
    return local_probs.float().mean(dim=(-2, -1))


def cross_view_confidence(s_glob, s_spat):
    s_glob = s_glob.float()
    s_spat = s_spat.float()
    if s_glob.shape != s_spat.shape:
        raise ValueError("score vectors must share shape")
    if s_glob.ndim != 1:
        raise ValueError("score vectors must be 1D [Q]")

    bar_g = _l1_positive(s_glob)
    bar_s = _l1_positive(s_spat)
    alpha = torch.sqrt(bar_g * bar_s).sum().clamp(0.0, 1.0)
    p = alpha * bar_g + (1.0 - alpha) * bar_s
    return p * (1.0 - _residual_entropy(p)), alpha


def _l1_positive(scores):
    positive = scores.clamp_min(0.0)
    total = positive.sum()
    if float(total) <= 0.0:
        return torch.full_like(scores, 1.0 / scores.numel())
    return positive / (total + EPS_NORM)


def _residual_entropy(p):
    count = p.numel()
    if count < 3:

        return torch.zeros_like(p)
    total_plogp = (p * torch.log(p.clamp_min(EPS_LOG))).sum()
    residual_mass = (1.0 - p).clamp_min(EPS_LOG)

    partial = total_plogp - p * torch.log(p.clamp_min(EPS_LOG))

    entropy = -(partial / residual_mass - torch.log(residual_mass))
    normalised = entropy / torch.log(torch.tensor(
        float(count - 1), device=p.device, dtype=p.dtype))
    normalised = torch.where(p >= 1.0, torch.zeros_like(normalised), normalised)
    return normalised.clamp(0.0, 1.0)
