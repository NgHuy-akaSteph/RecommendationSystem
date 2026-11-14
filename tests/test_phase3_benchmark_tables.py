"""Test các bảng benchmark chung CF/CB của phase 03.

Ba lỗi thật có thể xảy ra ở đúng những chỗ này và không test nào bắt được:

    - Bảng "cùng giao thức" lại lấy số liệu từ hai split khác nhau. Bảng CF/CB phải
      mang `split_id` và `catalog_id` giống nhau — đó là bằng chứng, không phải
      trang trí. Test ở đây kiểm tra chính điều kiện đó.
    - Cột `model` trong metrics CSV không khớp với mô hình mà bảng khai báo → bảng
      đang gộp hai thuật toán khác nhau mà không báo lỗi.
    - Wilcoxon ghép cặp giữa hai bảng khác nhau (CF với CB) là so sánh vô nghĩa vì
      hai bên được tinh chỉnh với số config khác nhau.
"""

import numpy as np
import pandas as pd
import pytest

from src import paths
from src.data.catalog_identity import catalog_id, split_id
from src.eval.benchmark_tables import (ALL_SPECS, CB_SPECS, CF_SPECS, build_table,
                                       result_name, wilcoxon_for_table)

# ── định danh catalog / split ───────────────────────────────────────────

def test_catalog_id_is_stable_and_size_sensitive():
    meta = pd.DataFrame({"movieId": [1, 2, 3]})
    assert catalog_id(meta) == catalog_id(meta)
    other = pd.DataFrame({"movieId": [1, 2, 4]})
    assert catalog_id(meta) != catalog_id(other)


def test_catalog_id_ignores_row_order():
    a = pd.DataFrame({"movieId": [3, 1, 2]})
    b = pd.DataFrame({"movieId": [1, 2, 3]})
    assert catalog_id(a) == catalog_id(b)


def test_split_id_distinguishes_main_from_coldstart():
    assert split_id(None) != split_id("coldstart")
    assert split_id("coldstart") == "temporal80_20+coldstart_item"


# ── tên kết quả ─────────────────────────────────────────────────────────

def test_result_name_matches_experiment_name(tmp_path):
    from src.eval.run_experiment_from_config import experiment_name
    for users, split_dir, expected in [
        ("dev", None, "x"),
        ("full", None, "full_x"),
        ("dev", tmp_path / "coldstart", "coldstart_x"),
    ]:
        assert result_name("x", users, split_dir) == experiment_name("x", users, split_dir)


# ── bảng benchmark ──────────────────────────────────────────────────────

def _write_metrics(rdir, exp_name, model, ndcg, split_id_="temporal80_20",
                   catalog_id_="ml25m-n13816-abc", n_users=20_000):
    pd.DataFrame([{
        "exp_name": exp_name, "model": model, "NDCG@10": ndcg,
        "NDCG@20": ndcg / 2, "HR@10": 0.5,
        "fit_s": 1.0, "score_s": 2.0, "n_eval_users": n_users,
        "split_id": split_id_, "catalog_id": catalog_id_, "seed": 42,
        "params": "{}", "users": "dev",
    }]).to_csv(rdir / f"{exp_name}_metrics.csv", index=False)


@pytest.fixture
def fake_results(tmp_path, monkeypatch):
    """results/ + configs/ giả lập cho đúng các mô hình trong registry."""
    rdir, cdir = tmp_path / "results", tmp_path / "configs"
    rdir.mkdir(); cdir.mkdir()
    for spec in ALL_SPECS:
        base = spec.config.replace(".yaml", "")
        (cdir / spec.config).write_text(
            f"exp_name: {base}\nmodel: {spec.model_key}\n", encoding="utf-8")
        _write_metrics(rdir, base, spec.model_key, 0.1)
    monkeypatch.setattr(paths, "RESULTS_DIR", rdir)
    monkeypatch.setattr(paths, "CONFIG_DIR", cdir)
    return rdir


def test_benchmark_table_has_all_registered_models(fake_results):
    """Bảng phải chứa **đúng** các mô hình trong registry — không thiếu, không thừa.

    Số lượng không hard-code: registry có thể thêm mô hình (CB-4 Hybrid được thêm sau
    khi sweep đủ), và test phải bắt được việc đó chứ không đòi người viết sửa magic
    number. Nhưng nó **có** bắt lỗi thật: một mô hình bị rơi khỏi bảng, hoặc một dòng
    thừa do cấu hình bị đăng nhập hai lần.
    """
    df = build_table("dev", ALL_SPECS, results_dir=fake_results)
    assert len(df) == len(ALL_SPECS)
    assert df["Mô hình"].nunique() == len(ALL_SPECS)
    assert set(df["Mô hình"]) == {s.label for s in ALL_SPECS}


def test_benchmark_table_carries_protocol_proof(fake_results):
    """Mọi dòng phải mang cùng split_id/catalog_id — bằng chứng cùng giao thức."""
    df = build_table("dev", ALL_SPECS, results_dir=fake_results)
    assert df["split_id"].nunique() == 1
    assert df["catalog_id"].nunique() == 1
    assert (df["seed"] == 42).all()


def test_cf_and_cb_tables_are_separate(fake_results):
    cf = build_table("dev", CF_SPECS, results_dir=fake_results)
    cb = build_table("dev", CB_SPECS, results_dir=fake_results)
    assert len(cf) == len(CF_SPECS)
    assert len(cb) == len(CB_SPECS)
    assert set(cf["Mô hình"]).isdisjoint(set(cb["Mô hình"]))
    # CF và CB phải là hai bảng tách biệt, không gộp lại thành một bảng xếp hạng
    assert set(CF_SPECS) & set(CB_SPECS) == set()


def test_benchmark_table_rejects_model_mismatch(fake_results):
    """Config đổi sau khi chạy → bảng phải báo lỗi chứ không gộp im lặng."""
    path = fake_results / "baseline-random_metrics.csv"
    df = pd.read_csv(path)
    df.loc[0, "model"] = "mostpop"
    df.to_csv(path, index=False)
    with pytest.raises(SystemExit, match="model"):
        build_table("dev", ALL_SPECS, results_dir=fake_results)


def test_benchmark_table_rejects_missing_result(fake_results):
    (fake_results / "baseline-random_metrics.csv").unlink()
    with pytest.raises(SystemExit, match="Thiếu kết quả"):
        build_table("dev", ALL_SPECS, results_dir=fake_results)


# ── Wilcoxon trong từng bảng ─────────────────────────────────────────────

def _write_peruser(rdir, exp_name, values, users=None):
    users = np.arange(len(values), dtype="int32") if users is None else users
    np.savez_compressed(rdir / f"{exp_name}_peruser.npz",
                        eval_users=users, **{"NDCG@10": np.asarray(values, "float32")})


def test_wilcoxon_only_compares_within_a_table(fake_results):
    """Wilcoxon của bảng CF chỉ được so trong CF — không lẫn CB."""
    for spec in ALL_SPECS:
        base = spec.config.replace(".yaml", "")
        _write_peruser(fake_results, base, np.linspace(0, 1, 100))
    wil = wilcoxon_for_table("dev", CF_SPECS, results_dir=fake_results)
    assert set(wil["Mô hình"]) <= {s.label for s in CF_SPECS}
    assert "mốc" in wil.columns


def test_wilcoxon_detects_different_user_sets(fake_results):
    """Hai mô hình chấm trên tập user khác nhau thì không ghép cặp được."""
    for spec in ALL_SPECS:
        base = spec.config.replace(".yaml", "")
        _write_peruser(fake_results, base, np.linspace(0, 1, 100))
    _write_peruser(fake_results, "baseline-random",
                   np.linspace(0, 1, 100), users=np.arange(1, 101, dtype="int32"))
    with pytest.raises(SystemExit, match="khác"):
        wilcoxon_for_table("dev", CF_SPECS, results_dir=fake_results)
