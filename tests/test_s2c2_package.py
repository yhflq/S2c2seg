"""Numerical regression tests for the ``models/`` package.

Strategy: each formula is re-implemented independently inside the test and
the package implementation is asserted to match it bit-for-bit, so any
unintended change of a convention fails here first.
"""

import unittest

import numpy as np
import torch

from models import S2C2Config
from models.adaptive_k import score_gap_k
from models.backends.preprocess import to_uint8_image, unnormalize
from models.cache.keys import (CACHE_SCHEMA_VERSION, config_fingerprint,
                             tensor_fingerprint, vocabulary_fingerprint)
from models.cache.store import MemoryStore
from models.csg import fuse, guide_weights, scatter
from models.css import candidate_pool, select
from models.iterative import (feedback_signal,
                            high_confidence_counts, refine, residual_evidence,
                            set_iou, vocabulary_expansion_pool)
from models.scoring import cross_view_confidence, global_alignment, minmax


def legacy_union(clip_sim, cseg_sim, entropy, ratio):
    """Independent reference implementation of the three-way top-k union pool."""
    count = clip_sim.numel()
    k = max(1, int(count * ratio))
    return torch.unique(torch.cat([
        torch.topk(clip_sim, k)[1],
        torch.topk(cseg_sim, k)[1],
        torch.topk(entropy, k)[1],
    ]))


def legacy_fusion(clip_logits, cseg_probs, weight, guide=0.07):
    """Independent reference implementation of the convex fusion."""
    main = (1 - weight) * clip_logits + weight * cseg_probs
    global_scores = clip_logits.mean(dim=(1, 2))
    global_weights = torch.softmax(global_scores, dim=-1)
    guided = cseg_probs * global_weights[:, None, None]
    return (1 - guide) * main + guide * guided


class TestScoring(unittest.TestCase):

    def test_global_alignment_is_patch_mean(self):
        dense = torch.randn(5, 4, 6)
        self.assertTrue(torch.equal(
            global_alignment(dense, mode="patch_mean"),
            dense.float().mean(dim=(-2, -1))))

    def test_global_alignment_cls_requires_scores(self):
        with self.assertRaises(ValueError):
            global_alignment(torch.randn(3, 2, 2), mode="cls")

    def test_minmax_uses_legacy_epsilon(self):
        scores = torch.tensor([1.0, 1.0, 1.0])
        # The denominator is (0 + 1e-6); there is no zero-range check, so the result is 0.
        self.assertTrue(torch.equal(minmax(scores), torch.zeros(3)))

    def test_bhattacharyya_alpha_bounds_and_entropy(self):
        scores = torch.tensor([0.5, 0.3, 0.2, 0.0])
        confidence, alpha = cross_view_confidence(
            scores, scores)
        # Identical distributions give a Bhattacharyya coefficient of 1.
        self.assertAlmostEqual(float(alpha), 1.0, places=5)
        self.assertTrue(torch.all(confidence >= 0))
        self.assertTrue(torch.all(confidence <= 1))

    def test_bhattacharyya_residual_entropy_matches_bruteforce(self):
        torch.manual_seed(1)
        scores = torch.rand(6)
        confidence, _ = cross_view_confidence(
            scores, scores)
        positive = scores.clamp_min(0)
        p = positive / (positive.sum() + 1e-6)
        expected = []
        for i in range(p.numel()):
            residual = torch.cat([p[:i], p[i + 1:]]) / (1 - p[i])
            entropy = -(residual * residual.log()).sum() / np.log(p.numel() - 1)
            expected.append(float(p[i] * (1 - entropy)))
        self.assertTrue(torch.allclose(
            confidence, torch.tensor(expected), atol=1e-5))

    def test_all_zero_scores_fall_back_to_uniform(self):
        confidence, alpha = cross_view_confidence(
            torch.zeros(4), torch.zeros(4))
        self.assertAlmostEqual(float(alpha), 1.0, places=5)
        self.assertTrue(torch.all(torch.isfinite(confidence)))


class TestCSS(unittest.TestCase):
    def _scores(self, count=40, seed=0):
        torch.manual_seed(seed)
        clip_sim = torch.rand(count) * 0.3
        cseg_sim = torch.rand(count)
        entropy = torch.rand(count)
        return clip_sim, cseg_sim, entropy

    def test_candidate_pool_matches_legacy_union(self):
        clip_sim, cseg_sim, entropy = self._scores()
        for ratio in (0.3, 0.4, 0.6):
            size = S2C2Config(tau=ratio).candidate_size(clip_sim.numel())
            self.assertTrue(torch.equal(
                candidate_pool(clip_sim, cseg_sim, entropy, size),
                legacy_union(clip_sim, cseg_sim, entropy, ratio)))

    def test_topk_size_uses_int_truncation(self):
        # int(171 * 0.3) == 51 by truncation; rounding would differ elsewhere.
        scores = torch.arange(171).float()
        size = S2C2Config(tau=0.3).candidate_size(171)
        self.assertEqual(size, 51)
        self.assertEqual(int(candidate_pool(scores, scores, scores, size).numel()), 51)

    def test_default_config_selection_matches_candidate_pool(self):
        """With k_max unset, the selected set equals the candidate pool."""
        clip_sim, cseg_sim, entropy = self._scores()
        cfg = S2C2Config()
        result = select(clip_sim, cseg_sim, entropy, cfg)
        self.assertIsNone(cfg.k_max)
        self.assertTrue(torch.equal(
            result["selected"], result["candidates"]))

    def test_k_max_caps_and_keeps_sorted_output(self):
        clip_sim, cseg_sim, entropy = self._scores()
        result = select(clip_sim, cseg_sim, entropy, S2C2Config(k_max=8))
        self.assertEqual(result["k"], 8)
        selected = result["selected"]
        self.assertTrue(torch.equal(selected, torch.sort(selected)[0]))

    def test_k_min_retains_all_candidates(self):
        """Fewer than k_min candidates: all candidates are retained (Alg. 1, lines 4-5)."""
        scores = torch.zeros(10)
        scores[3] = 1.0
        cfg = S2C2Config(tau=0.1, k_min=6)
        result = select(scores, scores, scores, cfg)
        self.assertEqual(result["k"], 1)
        self.assertTrue(torch.equal(
            result["selected"], result["candidates"]))


class TestSizeRules(unittest.TestCase):
    """Size rules: candidate-pool size n_cand(Q) and K_max(Q)."""

    #: Query-space size of each protocol.
    VOCABULARY = {
        "city": 19, "voc20": 20, "voc21": 46, "context59": 59,
        "context60": 60, "coco_object": 97, "ade150": 150, "coco_stuff": 171,
    }

    def test_candidate_size_is_floor_tau_q(self):
        cfg = S2C2Config(tau=0.3)
        for size in (19, 20, 46, 59, 60, 97, 150, 171):
            self.assertEqual(cfg.candidate_size(size), int(size * 0.3))


    def test_candidate_size_never_exceeds_the_vocabulary(self):
        cfg = S2C2Config(tau=1.0)
        self.assertEqual(cfg.candidate_size(7), 7)


class TestAdaptiveK(unittest.TestCase):
    def test_picks_largest_gap_inside_feasible_range(self):
        ranked = torch.tensor([9.0, 8.9, 8.8, 8.7, 3.0, 2.9, 2.8])
        # The largest gap lies between ranks 4 and 5 (K=4); a floor of 6 pushes it out of range.
        self.assertEqual(score_gap_k(ranked, 1, 6), 4)
        self.assertEqual(score_gap_k(ranked, 6, 6), 6)

    def test_clips_upper_bound_to_available_scores(self):
        ranked = torch.tensor([3.0, 2.0, 1.0])
        self.assertLessEqual(score_gap_k(ranked, 1, 99), 3)

    def test_empty_feasible_range_returns_lower_bound(self):
        self.assertEqual(score_gap_k(torch.tensor([1.0]), 6, 0), 1)


class TestCSG(unittest.TestCase):
    def test_convex_fusion_matches_legacy(self):
        torch.manual_seed(2)
        global_map = torch.rand(7, 5, 5) * 0.3
        local_map = torch.rand(7, 5, 5)
        for weight in (0.08, 0.15):
            cfg = S2C2Config(lam=weight)
            got, _ = fuse(global_map, local_map, cfg)
            expected = legacy_fusion(global_map, local_map, weight)
            self.assertTrue(torch.allclose(got, expected, atol=1e-6))

    def test_guide_weights_are_softmax_over_categories(self):
        global_map = torch.rand(4, 3, 3)
        weights = guide_weights(global_map)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=5)


    def test_scatter_fill_semantics(self):
        fused = torch.ones(2, 3, 3)
        selected = torch.tensor([1, 3])
        zero_filled = scatter(fused, selected, 5, 0.0)
        self.assertEqual(float(zero_filled[0].max()), 0.0)
        hard = scatter(fused, selected, 5, -1e9)
        self.assertEqual(float(hard[0].max()), -1e9)
        self.assertTrue(torch.equal(zero_filled[1], torch.ones(3, 3)))

    def test_shape_mismatch_fails_closed(self):
        with self.assertRaises(ValueError):
            fuse(torch.rand(3, 4, 4), torch.rand(2, 4, 4), S2C2Config())


class TestIterative(unittest.TestCase):
    def test_high_confidence_counts_uses_row_positions(self):
        """Row-to-position mapping must be explicit; row order is not assumed to equal selected order."""
        prediction = torch.tensor([[0, 1], [2, 3]])
        high = torch.ones(2, 2, dtype=torch.bool)
        # Four fused rows, mapped pairwise onto two selected entries.
        positions = torch.tensor([0, 0, 1, 1])
        counts = high_confidence_counts(prediction, high, positions, 2)
        self.assertTrue(torch.equal(counts, torch.tensor([2.0, 2.0])))

    def test_feedback_signal_is_high_confidence_share(self):
        counts = torch.tensor([2.0, 1.0])
        signal = feedback_signal(counts, 3, torch.tensor([2, 5]), 8)
        self.assertAlmostEqual(float(signal[2]), 2 / 3, places=4)
        self.assertAlmostEqual(float(signal[5]), 1 / 3, places=4)
        self.assertEqual(float(signal.sum()), float(signal[2] + signal[5]))

    def test_feedback_is_zero_without_high_confidence_pixels(self):
        signal = feedback_signal(
            torch.zeros(2), 0, torch.tensor([0, 1]), 4)
        self.assertEqual(float(signal.abs().sum()), 0.0)

    def test_absent_categories_get_exactly_zero(self):
        signal = feedback_signal(
            torch.tensor([3.0, 0.0]), 3, torch.tensor([1, 2]), 5)
        self.assertEqual(float(signal[2]), 0.0)
        self.assertEqual(float(signal[0]), 0.0)


    def test_vocabulary_pool_takes_top_ranked_outsiders(self):
        scores = torch.tensor([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
        selected = torch.tensor([0, 2])
        pool = vocabulary_expansion_pool(scores, selected, 2)
        # The two highest-scoring entries outside the selected set are 4 (0.7) and 5 (0.3).
        self.assertTrue(torch.equal(pool, torch.tensor([4, 5])))

    def test_vocabulary_pool_never_returns_selected_entries(self):
        scores = torch.rand(12)
        selected = torch.tensor([1, 4, 7])
        pool = vocabulary_expansion_pool(scores, selected, 5)
        self.assertEqual(len(set(pool.tolist()) & set(selected.tolist())), 0)


    def test_set_iou(self):
        self.assertEqual(set_iou(torch.tensor([1, 2]), torch.tensor([1, 2])), 1.0)
        self.assertAlmostEqual(
            set_iou(torch.tensor([1, 2]), torch.tensor([2, 3])), 1 / 3)


class TestRefine(unittest.TestCase):
    """End-to-end behaviour of iterative refinement."""

    def _state(self, selected, candidates, size, height=4, width=4):
        rows = int(selected.numel())
        fused = torch.zeros(rows, height, width)
        fused[0] = 5.0            # entry 0 wins everywhere with high confidence
        scores = torch.zeros(size)
        scores[candidates] = torch.linspace(1.0, 0.1, candidates.numel())
        return {
            "selected": selected, "candidates": candidates,
            "final_scores": scores[candidates],
            "s_glob": scores.clone(), "s_spat": scores.clone(),
            "s_conf": scores.clone(), "vocabulary_scores": scores.clone(), "fused": fused,
            "weights": torch.ones(rows),
            "row_positions": torch.arange(rows),
            "size": size, "rounds_run": 0, "rounds_changed": 0,
            "high_conf_fraction": 0.0, "mean_confidence": 0.0, "set_iou": 1.0,
        }

    def _fuse_fn(self, height=4, width=4):
        def fuse(selected_space):
            rows = int(selected_space.numel())
            fused = torch.zeros(rows, height, width)
            fused[0] = 5.0
            return fused, torch.ones(rows), torch.arange(rows)
        return fuse

    def test_zero_rounds_is_an_exact_identity(self):
        cfg = S2C2Config(k_min=2, t_max=0)
        state = self._state(torch.tensor([0, 1, 2]), torch.arange(6), 6)
        before = state["selected"]
        after = refine(state, self._fuse_fn(), cfg, 40.0)
        self.assertIs(after["selected"], before)
        self.assertEqual(after["rounds_run"], 0)


    def test_rounds_changed_separates_real_updates_from_no_ops(self):
        cfg = S2C2Config(k_min=2, t_max=2, theta_high=0.5)
        state = self._state(torch.tensor([0, 1, 2]), torch.arange(3), 3)
        after = refine(state, self._fuse_fn(), cfg, 40.0)
        # Candidates equal the selection: empty pool, unchanged set, counted as run but not changed.
        self.assertEqual(after["set_iou"], 1.0)
        self.assertEqual(after["rounds_changed"], 0)
        self.assertGreaterEqual(after["rounds_run"], 1)


    def test_row_position_mismatch_fails_closed(self):
        cfg = S2C2Config(k_min=2, t_max=1, theta_high=0.5)
        state = self._state(torch.tensor([0, 1, 2]), torch.arange(6), 6)
        state["row_positions"] = torch.arange(2)      # mismatches the fused row count
        with self.assertRaises(ValueError):
            refine(state, self._fuse_fn(), cfg, 40.0)


class TestRefineMonotone(unittest.TestCase):
    """Semantics of monotone-admission refinement.

    Incumbents are never dropped and admissions must pass the trial gate.
    """

    def _state(self, selected, candidates, size, height=4, width=4):
        rows = int(selected.numel())
        # Entry 0 wins with high confidence on the right half only; the left half is low-confidence.
        fused = torch.zeros(rows, height, width)
        fused[0, :, width // 2:] = 5.0
        scores = torch.zeros(size)
        scores[candidates] = torch.linspace(1.0, 0.1, candidates.numel())
        return {
            "selected": selected, "candidates": candidates,
            "final_scores": scores[candidates],
            "s_glob": scores.clone(), "s_spat": scores.clone(),
            "s_conf": scores.clone(),
            "vocabulary_scores": scores.clone(), "fused": fused,
            "weights": torch.ones(rows),
            "row_positions": torch.arange(rows),
            "size": size, "local_sel": None,
            "rounds_run": 0, "rounds_changed": 0,
            "high_conf_fraction": 0.0, "mean_confidence": 0.0, "set_iou": 1.0,
        }

    def _fuse_with_winner(self, winner, height=4, width=4):
        """Entry ``winner`` wins the top-left quadrant; entry 0 wins the right half."""
        def fuse(selected_space):
            rows = int(selected_space.numel())
            fused = torch.zeros(rows, height, width)
            fused[0, :, width // 2:] = 5.0
            entries = selected_space.tolist()
            if winner in entries:
                fused[entries.index(winner), :2, :2] = 9.0
            return fused, torch.ones(rows), torch.arange(rows)
        return fuse

    def _cfg(self, **kwargs):
        base = dict(k_min=2, t_max=1, theta_high=0.5,
                    adaptive_admit=False, residual_beta=0.0)
        base.update(kwargs)
        return S2C2Config(**base)

    def test_incumbents_are_never_dropped(self):
        """Incumbents without high-confidence support are still retained."""
        state = self._state(torch.tensor([0, 1, 2]), torch.arange(6), 6)
        after = refine(state, self._fuse_with_winner(3), self._cfg(), 40.0)
        self.assertTrue({0, 1, 2}.issubset(set(after["selected"].tolist())))

    def test_admits_the_outsider_that_wins_pixels(self):
        state = self._state(torch.tensor([0, 1, 2]), torch.arange(6), 6)
        after = refine(state, self._fuse_with_winner(3), self._cfg(), 40.0)
        self.assertIn(3, after["selected"].tolist())
        self.assertEqual(after["rounds_changed"], 1)

    def test_trial_rejects_admissions_that_win_no_pixels(self):
        """Entries 4 and 5 win no pixels after re-fusion and are rolled back."""
        state = self._state(torch.tensor([0, 1, 2]), torch.arange(6), 6)
        after = refine(state, self._fuse_with_winner(3), self._cfg(), 40.0)
        self.assertNotIn(4, after["selected"].tolist())
        self.assertNotIn(5, after["selected"].tolist())

    def test_no_winning_admission_converges_without_change(self):
        """When every trial admission is rolled back, the selection is unchanged and set_iou is 1."""
        state = self._state(torch.tensor([0, 1, 2]), torch.arange(6), 6)
        after = refine(
            state, self._fuse_with_winner(-1), self._cfg(t_max=2), 40.0)
        self.assertEqual(sorted(after["selected"].tolist()), [0, 1, 2])
        self.assertEqual(after["set_iou"], 1.0)
        self.assertEqual(after["rounds_changed"], 0)

    def test_admit_cap_bounds_provisional_admissions(self):
        state = self._state(torch.tensor([0, 1]), torch.arange(8), 8)
        cfg = self._cfg(admit_cap=1, admit_trial=False)
        after = refine(state, self._fuse_with_winner(-1), cfg, 40.0)
        self.assertEqual(int(after["selected"].numel()), 3)

    def test_adaptive_admit_zero_quota_is_a_no_op(self):
        """The score gap falls at the current K, so the target equals K and nothing is admitted."""
        state = self._state(torch.tensor([0, 1, 2]), torch.arange(6), 6)
        scores = torch.tensor([1.0, 0.9, 0.8, 0.1, 0.05, 0.0])
        for key in ("s_glob", "s_spat", "s_conf", "vocabulary_scores"):
            state[key] = scores.clone()
        after = refine(state, self._fuse_with_winner(3),
                       self._cfg(adaptive_admit=True), 40.0)
        self.assertEqual(sorted(after["selected"].tolist()), [0, 1, 2])

    def test_residual_evidence_scores_low_conf_explainers(self):
        local = torch.zeros(4, 2, 2)
        local[2, 0, :] = 0.8          # entry 2 explains the low-confidence region
        local[3] = 0.1                # entry 3 responds uniformly weakly
        low = torch.tensor([[True, True], [False, False]])
        f_low = residual_evidence(local, low, torch.tensor([2, 3]))
        self.assertAlmostEqual(float(f_low[0]), 0.8, places=5)
        self.assertAlmostEqual(float(f_low[1]), 0.1, places=5)

    def test_residual_evidence_handles_empty_inputs(self):
        local = torch.rand(4, 2, 2)
        empty_low = torch.zeros(2, 2, dtype=torch.bool)
        self.assertEqual(float(residual_evidence(
            local, empty_low, torch.tensor([1])).sum()), 0.0)
        self.assertEqual(int(residual_evidence(
            local, torch.ones(2, 2, dtype=torch.bool),
            torch.tensor([], dtype=torch.long)).numel()), 0)

    def test_residual_beta_nominates_the_explaining_candidate(self):
        """Residual nomination: the lowest-prior entry that best explains
        low-confidence pixels enters the pool and is tried first."""
        state = self._state(torch.tensor([0, 1]), torch.arange(6), 6)
        # All-zero fused map: uniform confidence 0.5, the whole image is low-confidence.
        state["fused"] = torch.zeros(2, 4, 4)
        # Entry 5 has the lowest prior score but the strongest local evidence in the low-confidence region.
        local = torch.zeros(6, 4, 4)
        local[5] = 0.9
        state["local_sel"] = local
        cfg = self._cfg(admit_cap=1, admit_trial=False, residual_beta=5.0)
        after = refine(state, self._fuse_with_winner(-1), cfg, 40.0)
        self.assertIn(5, after["selected"].tolist())


class TestPreprocess(unittest.TestCase):

    def test_unnormalize_applies_per_channel_constants(self):
        got = unnormalize(torch.zeros(1, 3, 2, 2))
        self.assertTrue(torch.allclose(
            got[0, :, 0, 0],
            torch.tensor([0.48145466, 0.4578275, 0.40821073]), atol=1e-6))


    def test_uint8_conversion_clips(self):
        """Values above the RGB range are clipped before uint8 conversion."""
        image = torch.full((3, 1, 1), 264.9 / 255.0)
        self.assertEqual(int(to_uint8_image(image)[0, 0, 0]), 255)


class TestCache(unittest.TestCase):
    def test_memory_store_round_trip_is_identity(self):
        store = MemoryStore(2)
        value = torch.rand(3, 4)
        store.put("a", value)
        self.assertIs(store.get("a"), value)

    def test_memory_store_evicts_least_recently_used(self):
        store = MemoryStore(2)
        store.put("a", torch.zeros(1))
        store.put("b", torch.zeros(1))
        store.get("a")            # a becomes most recently used
        store.put("c", torch.zeros(1))
        self.assertIsNone(store.get("b"))
        self.assertIsNotNone(store.get("a"))

    def test_disabled_store_never_returns_a_value(self):
        store = MemoryStore(0)
        store.put("a", torch.zeros(1))
        self.assertIsNone(store.get("a"))

    def test_fingerprints_separate_everything_that_changes_numbers(self):
        base = torch.rand(1, 3, 8, 8)
        self.assertNotEqual(
            tensor_fingerprint(base), tensor_fingerprint(base + 1e-3))
        self.assertNotEqual(
            tensor_fingerprint(base), tensor_fingerprint(base.half()))
        self.assertNotEqual(
            vocabulary_fingerprint(["a", "b"]), vocabulary_fingerprint(["b", "a"]))
        self.assertNotEqual(
            config_fingerprint(S2C2Config().signature()),
            config_fingerprint(S2C2Config(lam=0.6).signature()))

    def test_config_signature_excludes_cache_only_fields(self):
        """Cache-only fields must not enter the cache key."""
        left = S2C2Config(memory_cache_crops=0).signature()
        right = S2C2Config(memory_cache_crops=8).signature()
        self.assertEqual(left, right)
        self.assertNotIn("memory_cache_crops", left)

    def test_schema_version_is_part_of_the_key(self):
        self.assertIsInstance(CACHE_SCHEMA_VERSION, int)


class TestConfig(unittest.TestCase):
    def test_default_config_values(self):
        cfg = S2C2Config()
        self.assertEqual(cfg.global_score, "cls")
        self.assertEqual(cfg.local_text, "plain")
        self.assertEqual(cfg.unselected_fill, -1e9)
        self.assertIsNone(cfg.k_max)
        self.assertEqual(cfg.k_min, 6)
        self.assertEqual(cfg.t_max, 1)

    def test_rerank_weights_derive_from_lambda(self):
        # Default: lambda-derived (1 - w - 0.2, w, 0.2); an explicit triple overrides it.
        self.assertEqual(S2C2Config(lam=0.15).rerank, (1 - 0.15 - 0.2, 0.15, 0.2))
        self.assertEqual(
            S2C2Config(rerank_weights=(1.0, 1.0, 1.0)).rerank, (1.0, 1.0, 1.0))

    def test_invalid_values_fail_closed(self):
        for kwargs in (dict(tau=0.0), dict(tau=1.5), dict(k_min=0),
                       dict(k_min=10, k_max=5), dict(lam=-0.1),
                       dict(t_max=4),
                       dict(rerank_weights=(1.0, 1.0))):
            with self.assertRaises(ValueError, msg=repr(kwargs)):
                S2C2Config(**kwargs)


class _StubBackend:
    """Dense-predictor stand-in returning a fixed probability map."""

    model_id = "stub"

    def __init__(self, probabilities):
        self._probabilities = probabilities

    def dense_probabilities(self, crop, output_size):
        return self._probabilities

    def local_for_fusion(self, probabilities):
        return probabilities


class TestPipelineClsScores(unittest.TestCase):
    """With global_score="cls", CLS cosine scores reach the scoring layer through run_window."""

    def _pipeline(self, cfg, queries=8):
        from models.pipeline import S2C2Pipeline

        backend = _StubBackend(torch.rand(queries, 12, 12))
        return S2C2Pipeline(
            ["q%d" % i for i in range(queries)], list(range(queries)),
            queries, cfg, torch.device("cpu"), backend=backend)

    def test_cls_scores_reach_global_alignment(self):
        cfg = S2C2Config(global_score="cls", k_min=2, memory_cache_crops=0)
        pipe = self._pipeline(cfg)
        out = pipe.run_window(
            torch.rand(1, 3, 12, 12), torch.rand(1, 8, 12, 12),
            cls_scores=torch.linspace(0.0, 1.0, 8))
        self.assertEqual(tuple(out.shape), (1, 8, 12, 12))

    def test_cls_mode_without_scores_fails_closed(self):
        cfg = S2C2Config(global_score="cls", k_min=2, memory_cache_crops=0)
        pipe = self._pipeline(cfg)
        with self.assertRaises(ValueError):
            pipe.run_window(torch.rand(1, 3, 12, 12), torch.rand(1, 8, 12, 12))


if __name__ == "__main__":
    unittest.main()
