
import torch


def score_gap_k(ranked_scores, k_min, k_max):

    if ranked_scores.ndim != 1:
        raise ValueError("ranked_scores must be 1D")
    count = int(ranked_scores.numel())
    upper = min(int(k_max), count - 1)
    lower = int(k_min)
    if upper < lower:
        return min(lower, count)
    gaps = ranked_scores[lower - 1:upper] - ranked_scores[lower:upper + 1]
    return lower + int(torch.argmax(gaps))
