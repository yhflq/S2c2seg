import torch

from .css import composite_score
from .csg import confidence


def high_confidence_counts(prediction, high_mask, row_positions, size):
    if int(prediction.max()) >= row_positions.numel():
        raise ValueError(
            "row_positions covers %d rows but prediction indexes row %d"
            % (row_positions.numel(), int(prediction.max())))
    counts = torch.zeros(size, device=prediction.device, dtype=torch.float32)
    hits = prediction[high_mask]
    if hits.numel() == 0:
        return counts
    positions = row_positions[hits]
    return counts.index_add_(
        0, positions, torch.ones_like(positions, dtype=torch.float32))


def feedback_signal(counts, total, selected, size, eps=1e-6):
    signal = torch.zeros(size, device=selected.device, dtype=torch.float32)
    if total <= 0:
        return signal
    signal[selected] = counts / (total + eps)
    return signal


def vocabulary_expansion_pool(vocabulary_scores, selected, k_previous):
    order = torch.argsort(vocabulary_scores, descending=True)
    outside = order[torch.isin(order, selected, invert=True)]
    take = min(int(k_previous), int(outside.numel()))
    return outside[:take]


def set_iou(left, right):
    left_set = set(int(index) for index in left.tolist())
    right_set = set(int(index) for index in right.tolist())
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def residual_evidence(local_sel, low_mask, entries, eps=1e-6):
    if int(entries.numel()) == 0 or not bool(low_mask.any()):
        return torch.zeros(
            int(entries.numel()), device=entries.device, dtype=torch.float32)
    return local_sel[entries][:, low_mask].float().mean(dim=1)


def refine(state, fuse_fn, cfg, logit_scale):
    selected = state["selected"]
    fused = state["fused"]
    positions = state["row_positions"]
    size = state["size"]
    rounds_run = 0
    rounds_changed = 0
    high_fraction = 0.0
    mean_confidence = 0.0
    last_iou = 1.0

    for _ in range(int(cfg.t_max)):
        if fused.shape[0] != positions.numel():
            raise ValueError(
                "fused has %d rows but row_positions covers %d"
                % (fused.shape[0], positions.numel()))
        scores, prediction = confidence(fused, logit_scale)
        high_mask = scores > cfg.theta_high
        high_count = int(high_mask.sum())
        high_fraction = high_count / high_mask.numel()
        mean_confidence = float(scores.mean())
        rounds_run += 1

        k_previous = int(selected.numel())
        counts = high_confidence_counts(
            prediction, high_mask, positions, k_previous)
        signal = feedback_signal(counts, high_count, selected, size)

        pool = vocabulary_expansion_pool(
            state["vocabulary_scores"], selected, k_previous)

        refined_glob = state["s_glob"] + cfg.beta * signal
        if cfg.residual_beta > 0 and state.get("local_sel") is not None:

            outside_all = torch.ones(
                size, dtype=torch.bool, device=selected.device)
            outside_all[selected] = False
            outsiders = outside_all.nonzero().flatten()
            f_low = residual_evidence(
                state["local_sel"], ~high_mask, outsiders)
            boost = torch.zeros(
                size, device=refined_glob.device, dtype=torch.float32)
            boost[outsiders] = f_low
            refined_glob = refined_glob + cfg.residual_beta * boost
            if int(outsiders.numel()) > 0:
                extra = min(int(cfg.admit_cap), int(outsiders.numel()))
                top_low = outsiders[torch.topk(f_low, extra)[1]]
                pool = torch.unique(torch.cat([pool, top_low]))
        if int(pool.numel()) == 0:
            break

        candidate_set = torch.unique(torch.cat([selected, pool]))
        refined_scores = composite_score(
            refined_glob, state["s_spat"], state["s_conf"],
            cfg.rerank, candidate_set)

        if cfg.adaptive_admit:

            from .adaptive_k import score_gap_k

            upper = int(candidate_set.numel()) - 1
            if upper < cfg.k_min:
                quota = 0
            else:
                ranked = torch.sort(refined_scores, descending=True)[0]
                target = score_gap_k(ranked, cfg.k_min, upper)
                quota = max(0, int(target) - k_previous)
        else:
            quota = int(pool.numel())
        quota = min(quota, int(cfg.admit_cap))
        if quota <= 0:
            break

        outside = torch.isin(candidate_set, selected, invert=True)
        pool_scores = refined_scores[outside]
        pool_indices = candidate_set[outside]
        take = min(quota, int(pool_indices.numel()))
        order = torch.topk(pool_scores, take)[1]
        provisional = pool_indices[order]

        updated = torch.sort(torch.unique(
            torch.cat([selected, provisional])))[0]
        fused_new, weights_new, positions_new = fuse_fn(updated)

        if cfg.admit_trial:

            low_mask = ~high_mask
            denom = int(low_mask.sum())
            if denom == 0:
                kept = provisional.new_empty((0,))
            else:
                _, prediction_new = confidence(fused_new, logit_scale)
                counts_new = high_confidence_counts(
                    prediction_new, low_mask, positions_new,
                    int(updated.numel()))
                pos_of = torch.searchsorted(updated, provisional)
                kept = provisional[
                    counts_new[pos_of] > 0]
        else:
            kept = provisional

        if int(kept.numel()) == 0:
            last_iou = 1.0
            break
        if int(kept.numel()) != int(provisional.numel()):
            updated = torch.sort(torch.unique(
                torch.cat([selected, kept])))[0]
            fused_new, weights_new, positions_new = fuse_fn(updated)

        last_iou = set_iou(updated, selected)
        rounds_changed += 1
        selected, fused, positions = updated, fused_new, positions_new
        state["weights"] = weights_new
        if last_iou >= 1.0 - cfg.delta:
            break

    state["selected"] = selected
    state["fused"] = fused
    state["row_positions"] = positions
    state["rounds_run"] = rounds_run
    state["rounds_changed"] = rounds_changed
    state["high_conf_fraction"] = high_fraction
    state["mean_confidence"] = mean_confidence
    state["set_iou"] = last_iou
    return state
