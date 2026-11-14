"""Gom mọi lưới thí nghiệm thành một điểm truy cập duy nhất."""

from pathlib import Path

from ablation_grids_content_models import cb1_configs, cb2_configs, cb3_configs
from ablation_grids_fusion_and_cf import (
    bm25_fusion_configs, cf_configs, hybrid_pop_configs,
    late_fusion_configs, pop_prior_configs, time_decay_configs,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"


def all_configs() -> list[tuple[str, str, dict]]:
    """Toàn bộ thí nghiệm, theo thứ tự trình bày trong báo cáo."""
    return (cb1_configs() + cb2_configs() + cb3_configs() + cf_configs()
            + pop_prior_configs() + late_fusion_configs() + hybrid_pop_configs()
            + bm25_fusion_configs() + time_decay_configs())
