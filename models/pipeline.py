import time

import torch

from . import css as css_module
from . import csg as csg_module
from . import scoring
from .backends.clipseg import CLIPSegBackend
from .cache.keys import (config_fingerprint, crop_key, tensor_fingerprint,
                         vocabulary_fingerprint)
from .cache.store import MemoryStore
from .iterative import refine

class S2C2Pipeline:

    def __init__(self, query_words, query_to_class, num_classes, cfg,
                 device, logit_scale=40.0, backend=None):
        self.cfg = cfg
        self.last_selected = None

        self.device = device
        self.logit_scale = float(logit_scale)
        self.query_words = list(query_words)
        self.num_queries = len(self.query_words)
        self.num_classes = int(num_classes)
        self.query_to_class = torch.as_tensor(
            query_to_class, dtype=torch.long, device=device)
        self.backend = backend if backend is not None \
            else CLIPSegBackend(self.query_words, cfg, device)

        self._vocabulary_hash = vocabulary_fingerprint(self.query_words)

        backend_fields = ("clipseg_model", "clipseg_query_batch", "local_text")
        self._backend_hash = config_fingerprint(
            {name: getattr(cfg, name) for name in backend_fields},
            extra={"clipseg": getattr(self.backend, "model_id", "injected")})
        self._memory = MemoryStore(cfg.memory_cache_crops)
        self.stats = _new_stats()

    @torch.no_grad()
    def run_window(self, normalized_crop, dense_query_logits, cls_scores=None):
        started = time.perf_counter()
        if dense_query_logits.shape[0] != 1:
            raise ValueError("S2C2 pipeline requires batch size 1")
        dense = dense_query_logits[0].float()
        if dense.shape[0] != self.num_queries:
            raise ValueError(
                "dense map has %d channels but the vocabulary has %d queries"
                % (dense.shape[0], self.num_queries))
        output_size = tuple(dense.shape[-2:])

        probabilities = self._local_probabilities(normalized_crop, output_size)
        local = probabilities

        s_glob = scoring.global_alignment(
            dense, mode=self.cfg.global_score, cls_scores=cls_scores)
        s_spat = scoring.spatial_presence(probabilities)
        space_glob, space_spat = s_glob, s_spat
        s_conf, alpha = scoring.cross_view_confidence(
            space_glob, space_spat)

        selection = css_module.select(space_glob, space_spat, s_conf, self.cfg)
        queries = selection["selected"]

        def fuse_fn(selected_space):
            picked = selected_space
            mapped, _ = csg_module.fuse(
                dense[picked], local[picked], self.cfg)
            return mapped, _, torch.searchsorted(picked, selected_space)

        self.last_selected = selection["selected"]
        fused, weights = csg_module.fuse(dense[queries], local[queries], self.cfg)

        state = {
            "selected": selection["selected"],
            "candidates": selection["candidates"],
            "final_scores": selection["final_scores"],
            "s_glob": space_glob, "s_spat": space_spat, "s_conf": s_conf,
            "vocabulary_scores": css_module.composite_score(
                space_glob, space_spat, s_conf, self.cfg.rerank),
            "fused": fused, "weights": weights,
            "row_positions": torch.searchsorted(
                queries, selection["selected"]),
            "size": int(space_glob.numel()),
            "local_sel": self._selection_local(probabilities),
            "rounds_run": 0, "rounds_changed": 0, "high_conf_fraction": 0.0,
            "mean_confidence": 0.0, "set_iou": 1.0,
        }
        if self.cfg.t_max > 0:
            state = refine(state, fuse_fn, self.cfg, self.logit_scale)
            queries = state["selected"]

        full = csg_module.scatter(
            state["fused"], queries, self.num_queries,
            self.cfg.unselected_fill, device=dense.device)

        self._record(state, selection, alpha, time.perf_counter() - started)
        return full.unsqueeze(0)

    def _local_probabilities(self, normalized_crop, output_size):
        key = None
        if self._memory.capacity > 0:
            key = crop_key(tensor_fingerprint(normalized_crop),
                           self._backend_hash, self._vocabulary_hash)
            cached = self._memory.get(key)
            if cached is not None and tuple(cached.shape[-2:]) == output_size:
                return cached
        probabilities = self.backend.dense_probabilities(
            normalized_crop, output_size)
        if key is not None:
            self._memory.put(key, probabilities)
        return probabilities

    def _selection_local(self, probabilities):
        if self.cfg.t_max > 0 and self.cfg.residual_beta > 0:
            return probabilities
        return None

    def _record(self, state, selection, alpha, seconds):
        stats = self.stats
        stats["windows"] += 1
        stats["selected_k"] += int(state["selected"].numel())
        stats["candidate_k"] += selection["candidate_count"]
        stats["rounds_run"] += state["rounds_run"]
        stats["rounds_changed"] += state.get("rounds_changed", 0)
        stats["mean_confidence"] += state.get("mean_confidence", 0.0)
        stats["high_conf_fraction"] += state["high_conf_fraction"]
        stats["set_iou"] += state["set_iou"]
        stats["seconds"] += seconds
        if torch.isfinite(alpha):
            stats["alpha"] += float(alpha)
            stats["alpha_windows"] += 1

    def evaluation_metrics(self):
        stats = self.stats
        windows = max(1, stats["windows"])
        metrics = {
            "s2c2Windows": float(stats["windows"]),
            "selectedK": stats["selected_k"] / windows,
            "candidateK": stats["candidate_k"] / windows,
            "roundsRun": stats["rounds_run"] / windows,
            "roundsChanged": stats["rounds_changed"] / windows,
            "confMean": stats["mean_confidence"] / windows,
            "highConfFrac": stats["high_conf_fraction"] / windows,
            "setIoU": stats["set_iou"] / windows,
            "s2c2Seconds": stats["seconds"],
        }
        if stats["alpha_windows"]:
            metrics["cssAlpha"] = stats["alpha"] / stats["alpha_windows"]
        memory = self._memory.stats()
        metrics["cacheHitRate"] = memory["hit_rate"]
        metrics["cacheHits"] = float(memory["hits"])
        return metrics

    def reset_stats(self):
        self.stats = _new_stats()

def _new_stats():
    return {
        "windows": 0, "selected_k": 0, "candidate_k": 0, "rounds_run": 0,
        "rounds_changed": 0, "mean_confidence": 0.0,
        "high_conf_fraction": 0.0, "set_iou": 0.0, "seconds": 0.0,
        "alpha": 0.0, "alpha_windows": 0,
    }
