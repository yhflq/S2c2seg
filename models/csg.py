import torch


def guide_weights(global_map):
    return torch.softmax(global_map.float().mean(dim=(-2, -1)), dim=-1)


def fuse(global_map, local_map, cfg):
    if global_map.shape != local_map.shape:
        raise ValueError("global and local maps must share shape, got %r and %r"
                         % (tuple(global_map.shape), tuple(local_map.shape)))
    if global_map.ndim != 3:
        raise ValueError("maps must be [K, H, W]")

    global_map = global_map.float()
    local_map = local_map.float()
    lam = float(cfg.lam)

    main = (1.0 - lam) * global_map + lam * local_map
    weights = guide_weights(global_map)
    guided = local_map * weights[:, None, None]
    gw = float(cfg.guide_weight)
    return (1.0 - gw) * main + gw * guided, weights


def scatter(fused, selected, size, fill, dtype=torch.float32, device=None):
    if device is None:
        device = fused.device
    full = torch.full((size,) + tuple(fused.shape[-2:]), float(fill),
                      dtype=dtype, device=device)
    full[selected] = fused.to(dtype)
    return full


def confidence(fused, logit_scale):
    probabilities = (fused.float() * float(logit_scale)).softmax(dim=0)
    return probabilities.max(dim=0)
