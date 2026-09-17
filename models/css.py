import torch

from .scoring import minmax


def candidate_pool(s_glob, s_spat, s_conf, size):
    count = s_glob.numel()
    top_k = max(1, int(size))
    picks = [
        torch.topk(scores, min(top_k, count))[1]
        for scores in (s_glob, s_spat, s_conf)
    ]
    return torch.unique(torch.cat(picks))


def composite_score(s_glob, s_spat, s_conf, weights, index=None):
    if index is not None:
        s_glob, s_spat, s_conf = s_glob[index], s_spat[index], s_conf[index]
    w_glob, w_spat, w_conf = weights
    return (w_glob * minmax(s_glob)
            + w_spat * minmax(s_spat)
            + w_conf * minmax(s_conf))


def select(s_glob, s_spat, s_conf, cfg):
    vocabulary_size = int(s_glob.numel())
    pool_size = cfg.candidate_size(vocabulary_size)
    candidates = candidate_pool(s_glob, s_spat, s_conf, pool_size)
    candidate_count = int(candidates.numel())
    weights = cfg.rerank

    final_scores = composite_score(s_glob, s_spat, s_conf, weights, candidates)
    selected = candidates


    limit = None
    if candidate_count >= cfg.k_min:

        limit = cfg.k_max
        if limit is not None and candidate_count > limit:
            order = torch.topk(final_scores, limit)[1]
            selected = candidates[order]

    selected = torch.sort(selected)[0]
    return {
        "selected": selected,
        "candidates": candidates,
        "final_scores": final_scores,
        "k": int(selected.numel()),
        "candidate_count": candidate_count,
        "k_max": limit,
    }
