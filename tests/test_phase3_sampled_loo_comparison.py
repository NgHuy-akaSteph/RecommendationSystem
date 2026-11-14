"""Test script so sánh giao thức sampled LOO (`scripts/run_sampled_loo_comparison.py`).

Script này từng tính **Kendall τ sai** vì hai lỗi độc lập:

    1. **Đọc sai file.** Bản cũ đọc `results/dev-summary.csv` — artifact trung gian bị
       gitignore, thiếu cột `split_id`/`catalog_id`/`seed`, và **không có**
       `cb4-04-late-bm25-w075`. Bản `results/final/dev-summary.csv` có đủ cột bằng
       chứng protocol nhưng chỉ 9 dòng và cũng thiếu `cb4-04-late-bm25-w075`.
       → Dùng `results/{exp_name}_metrics.csv` làm nguồn chuẩn.

    2. **Trùng `exp_name`.** `best-rs-itemcf.yaml` và `best-rs-itemcf-binary-pos.yaml`
       cùng trỏ tới `rs-itemcf-k200-binary-pos` → merge sinh hai dòng trùng →
       `rank()` tính sai → Kendall τ sai.
       → Dedup theo `exp_name` trước khi chạy.
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src import paths

ROOT = paths.ROOT
SCRIPT = ROOT / "scripts" / "run_sampled_loo_comparison.py"


def _load_module():
    """Nạp script dưới dạng module để test được hàm bên trong."""
    spec = importlib.util.spec_from_file_location("run_sampled_loo_comparison", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def loo():
    return _load_module()


# ── nguồn metric ─────────────────────────────────────────────────────────

def test_script_does_not_read_stale_dev_summary():
    """Script không được đọc bảng tổng hợp trung gian `results/dev-summary.csv`.

    File đó bị gitignore, thiếu bằng chứng protocol, và không chứa
    `cb4-04-late-bm25-w075` — đọc nó sẽ âm thầm mất dòng.
    """
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'paths.RESULTS_DIR / "dev-summary.csv"' not in text
    assert '"dev-summary.csv"' not in text


def test_script_reads_per_experiment_metric_files(loo):
    """Nguồn metric phải là `results/{exp_name}_metrics.csv` — đủ 12 exp_name."""
    assert loo.METRIC_TEMPLATE == "metrics.csv"

    configs = loo.unique_best_configs()
    missing = []
    for config in configs:
        exp_name = loo._read_exp_name(config)
        if not (paths.RESULTS_DIR / f"{exp_name}_metrics.csv").exists():
            missing.append(exp_name)
    assert not missing, f"thiếu file metric cho: {missing}"


# ── dedup theo exp_name ──────────────────────────────────────────────────

def test_unique_best_configs_drops_duplicate_exp_name(loo, tmp_path):
    """Hai config cùng `exp_name` phải bị gộp còn một, và **không mất** config nào."""
    def _write(name, exp_name):
        (tmp_path / name).write_text(
            yaml.safe_dump({"exp_name": exp_name, "model": "x"}), encoding="utf-8"
        )

    _write("best-aaa.yaml", "exp-1")
    _write("best-bbb.yaml", "exp-2")
    _write("best-ccc.yaml", "exp-2")   # trùng với best-bbb

    kept = loo.unique_best_configs(tmp_path)
    names = [loo._read_exp_name(p) for p in kept]
    assert sorted(names) == ["exp-1", "exp-2"]
    assert len(names) == len(set(names))
    # Giữ file đứng trước theo sorted() → "best-bbb.yaml"
    assert kept[1].name == "best-bbb.yaml"


def test_unique_best_configs_keeps_all_distinct_exp_names(loo):
    """Toàn bộ config `best-*.yaml` của repo phải cho 12 `exp_name` khác nhau."""
    kept = loo.unique_best_configs()
    names = [loo._read_exp_name(p) for p in kept]
    assert len(names) == len(set(names)), f"vẫn còn trùng: {names}"
    # 13 file `best-*.yaml` nhưng 12 `exp_name` (best-rs-itemcf.yaml trùng
    # best-rs-itemcf-binary-pos.yaml)
    assert len(kept) == 12


def test_real_configs_contain_the_known_duplicate(loo):
    """Bằng chứng lỗi trùng `exp_name` vẫn còn trong repo (chứ không phải giả định)."""
    all_configs = sorted(paths.CONFIG_DIR.glob("best-*.yaml"))
    names = [loo._read_exp_name(p) for p in all_configs]
    assert len(names) > len(set(names)), "repo không còn config trùng exp_name nào"
    assert "rs-itemcf-k200-binary-pos" in names


# ── bảng thứ hạng full-ranking ──────────────────────────────────────────

def test_load_full_ranking_returns_one_row_per_config(loo):
    """Bảng full-ranking phải có đúng 12 dòng, 12 exp_name khác nhau."""
    configs = loo.unique_best_configs()
    full = loo.load_full_ranking(configs)
    assert len(full) == 12
    assert full["exp_name"].nunique() == 12
    assert "config" in full.columns
    assert {"NDCG@10_full", "split_id", "catalog_id", "seed"} <= set(full.columns)


def test_load_full_ranking_includes_cb4_04(loo):
    """`cb4-04-late-bm25-w075` là dòng mà bảng `dev-summary.csv` thiếu — phải có mặt."""
    configs = loo.unique_best_configs()
    full = loo.load_full_ranking(configs)
    assert "cb4-04-late-bm25-w075" in set(full["exp_name"])


def test_load_full_ranking_raises_when_metric_missing(loo, tmp_path, monkeypatch):
    """Thiếu file metric thì phải báo lỗi rõ ràng, không âm thầm bỏ dòng."""
    config = tmp_path / "best-x.yaml"
    config.write_text(yaml.safe_dump({"exp_name": "no-such-exp", "model": "x"}),
                      encoding="utf-8")
    with pytest.raises(SystemExit, match="Thiếu file metric"):
        loo.load_full_ranking([config])


# ── merge: không trùng, không mất ────────────────────────────────────────

def _fake_sampled(n=12):
    """Bảng sampled giả với đủ cột mà `compare_protocols` cần."""
    rows = []
    for i in range(n):
        rows.append({
            "exp_name": f"exp-{i}",
            "model": "m",
            "config": f"best-{i}.yaml",
            "NDCG@10": 0.1 - i * 0.001,
            "HR@10": 0.5 - i * 0.01,
        })
    return pd.DataFrame(rows)


def _fake_full(n=12):
    return pd.DataFrame({
        "exp_name": [f"exp-{i}" for i in range(n)],
        "config": [f"best-{i}.yaml" for i in range(n)],
        "NDCG@10_full": [0.2 - i * 0.002 for i in range(n)],
        "split_id": ["temporal80_20"] * n,
        "catalog_id": ["ml25m-n13816"] * n,
        "seed": [42] * n,
    })


def test_compare_protocols_keeps_all_rows(loo):
    """Merge theo cặp khoá (`exp_name`, `config`) phải giữ đủ dòng."""
    merged = loo.compare_protocols(_fake_sampled(), _fake_full())
    assert len(merged) == 12
    assert merged["exp_name"].nunique() == 12
    assert "hạng_full" in merged.columns and "hạng_sampled" in merged.columns


def test_compare_protocols_rejects_duplicate_rows(loo):
    """Bảng sampled có dòng trùng `exp_name` phải bị chặn, không âm thầm tính sai τ."""
    sampled = _fake_sampled()
    dup = sampled.iloc[[0]].copy()
    dup["config"] = "best-dup.yaml"
    sampled = pd.concat([sampled, dup], ignore_index=True)

    full = _fake_full()
    with pytest.raises(SystemExit, match="trùng"):
        loo.compare_protocols(sampled, full)


def test_compare_protocols_rejects_missing_rows(loo):
    """Bảng sampled thiếu dòng phải bị chặn — đó là lỗi làm mất `cb4-04` trước đây."""
    sampled = _fake_sampled()[:-1]
    with pytest.raises(SystemExit, match="Merge sai"):
        loo.compare_protocols(sampled, _fake_full())


def test_compare_protocols_ranks_are_dense_and_ordered(loo):
    """Hạng phải là 1..n không trùng, và sắp theo hạng full."""
    merged = loo.compare_protocols(_fake_sampled(), _fake_full())
    assert sorted(merged["hạng_full"]) == list(range(1, 13))
    assert sorted(merged["hạng_sampled"]) == list(range(1, 13))
    assert list(merged["hạng_full"]) == sorted(merged["hạng_full"])
