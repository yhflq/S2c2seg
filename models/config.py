from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class S2C2Config:
    """Shared journal settings; scripts specify protocol-specific overrides."""

    tau: float = 0.3
    k_min: int = 6
    k_max: Optional[int] = None
    lam: float = 0.15
    t_max: int = 1
    beta: float = 0.5
    theta_high: float = 0.8
    delta: float = 0.1
    residual_beta: float = 0.5
    admit_cap: int = 3
    admit_trial: bool = True
    adaptive_admit: bool = True
    global_score: str = "cls"
    unselected_fill: float = -1e9
    rerank_weights: Optional[Tuple[float, float, float]] = None
    rerank_conf_weight: float = 0.20
    guide_weight: float = 0.07
    local_text: str = "plain"
    clipseg_model: str = "rd64-refined"
    clipseg_query_batch: int = 0
    memory_cache_crops: int = 8

    def __post_init__(self):
        if not 0 < self.tau <= 1:
            raise ValueError("tau must lie in (0, 1]")
        if self.k_min < 1:
            raise ValueError("k_min must be >= 1")
        if self.k_max is not None and self.k_max < self.k_min:
            raise ValueError("require k_min <= k_max")
        if not 0 <= self.lam <= 1:
            raise ValueError("lam must lie in [0, 1] under the convex parametrisation")
        if not 0 <= self.t_max <= 3:
            raise ValueError("t_max must lie in [0, 3]")
        if not 0 <= self.theta_high <= 1:
            raise ValueError("theta_high must lie in [0, 1]")
        if not 0 <= self.delta <= 1:
            raise ValueError("delta must lie in [0, 1]")
        if not 0 <= self.guide_weight <= 1:
            raise ValueError("guide_weight must lie in [0, 1]")
        _choice("global_score", self.global_score, {"patch_mean", "cls"})
        _choice("local_text", self.local_text, {"plain", "template"})
        if self.residual_beta < 0:
            raise ValueError("residual_beta must be >= 0")
        if self.admit_cap < 1:
            raise ValueError("admit_cap must be >= 1")
        if self.rerank_weights is not None and len(self.rerank_weights) != 3:
            raise ValueError("rerank_weights must be a 3-tuple (glob, spat, conf)")
        if self.memory_cache_crops < 0:
            raise ValueError("memory_cache_crops must be >= 0")

    @property
    def rerank(self) -> Tuple[float, float, float]:
        if self.rerank_weights is not None:
            return tuple(float(w) for w in self.rerank_weights)
        eps = float(self.rerank_conf_weight)
        return (1.0 - float(self.lam) - eps, float(self.lam), eps)

    def candidate_size(self, vocabulary_size: int) -> int:
        return max(1, min(int(vocabulary_size * self.tau), int(vocabulary_size)))

    def signature(self) -> dict:
        excluded = {"memory_cache_crops"}
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in excluded
        }


def _choice(name, value, allowed):
    if value not in allowed:
        raise ValueError(
            "%s must be one of %s, got %r" % (name, sorted(allowed), value))
