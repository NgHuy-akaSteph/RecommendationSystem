"""Test các hàng rào nối phase 3 với phase 4.

Hai lỗi thật đã xảy ra ở đúng những chỗ này và không test nào bắt được:
    - `collect()` gộp cả kết quả full_/coldstart_ vào phép chọn cấu hình tốt nhất, khiến
      `select_best_configs` trỏ sang exp_name không có file config và **im lặng** giữ
      nguyên best-*.yaml cũ, đồng thời ghi số liệu full vào file tên dev-summary.
    - `bucket_table` tính `sizes` từ file đầu tiên rồi áp cho mọi file sau mà không kiểm
      tra chúng có cùng tập user không.
"""

import numpy as np
import pandas as pd
import pytest

from src import paths
from src.eval.collect_results_to_table import collect, strip_protocol_prefix
from src.eval.run_experiment_from_config import experiment_name
from src.eval.statistical_tests import load_all
from src.eval.user_bucket_analysis import bucket_table, paired_bucket_diffs


# ── tiền tố giao thức ────────────────────────────────────────────────────

@pytest.mark.parametrize("name,expected", [
    ("cb1-01-add-tags-tfidf", "cb1-01-add-tags-tfidf"),
    ("full_cb1-01-add-tags-tfidf", "cb1-01-add-tags-tfidf"),
    ("coldstart_cb1-01-add-tags-tfidf", "cb1-01-add-tags-tfidf"),
    ("coldstart_full_cb1-01", "cb1-01"),
    ("fullstack-model", "fullstack-model"),      # không được cắt nhầm tên hợp lệ
    ("coldstart-baseline", "coldstart-baseline"),
])
def test_strip_protocol_prefix(name, expected):
    assert strip_protocol_prefix(name) == expected


def test_experiment_name_round_trips_with_strip(tmp_path):
    assert experiment_name("x", "dev", None) == "x"
    assert experiment_name("x", "full", None) == "full_x"
    assert strip_protocol_prefix(experiment_name("x", "full", tmp_path / "coldstart")) == "x"


# ── collect(protocol=...) ────────────────────────────────────────────────

def _write_metrics(rdir, exp_name, ndcg, model="cb1-tfidf"):
    pd.DataFrame([{"exp_name": exp_name, "model": model, "NDCG@10": ndcg,
                   "params": "{}", "users": "dev"}]).to_csv(
        rdir / f"{exp_name}_metrics.csv", index=False)


@pytest.fixture
def results_with_protocols(tmp_path, monkeypatch):
    """results/ có cả dev lẫn full; configs/ chỉ có file config của dòng dev."""
    rdir, cdir = tmp_path / "results", tmp_path / "configs"
    rdir.mkdir(); cdir.mkdir()
    _write_metrics(rdir, "cb1-01", 0.09)
    _write_metrics(rdir, "full_cb1-01", 0.12)        # điểm full cao hơn dev
    (cdir / "cb1-01.yaml").write_text("exp_name: cb1-01\n", encoding="utf-8")
    monkeypatch.setattr(paths, "RESULTS_DIR", rdir)
    monkeypatch.setattr(paths, "CONFIG_DIR", cdir)
    return rdir


def test_collect_all_includes_protocol_runs(results_with_protocols):
    names = set(collect(sort_by=None)["exp_name"])
    assert names == {"cb1-01", "full_cb1-01"}


def test_collect_dev_excludes_protocol_runs(results_with_protocols):
    assert set(collect(sort_by=None, protocol="dev")["exp_name"]) == {"cb1-01"}


def test_collect_rejects_unknown_protocol(results_with_protocols):
    with pytest.raises(ValueError, match="protocol"):
        collect(sort_by=None, protocol="bogus")


def test_select_best_never_picks_a_protocol_run(results_with_protocols):
    """Chốt lỗi C1: argmax không lọc sẽ chọn full_cb1-01 (0,12) thay vì cb1-01 (0,09).

    Hệ quả nếu để lọt: write_best_configs đi tìm configs/full_cb1-01.yaml, không thấy,
    in cảnh báo rồi **giữ nguyên** best-*.yaml cũ, và dev-summary.csv chứa số liệu full.
    """
    import sys
    sys.path.insert(0, str(paths.ROOT / "scripts"))
    import select_best_configs

    best = select_best_configs.select_best("NDCG@10")
    assert list(best["exp_name"]) == ["cb1-01"]
    assert float(best["NDCG@10"].iloc[0]) == pytest.approx(0.09)


def test_orphan_guard_still_drops_stale_results(results_with_protocols, capsys):
    """Kết quả không còn config tương ứng vẫn phải bị loại, kể cả khi có tiền tố."""
    _write_metrics(results_with_protocols, "cb1-99-da-xoa", 0.5)
    names = set(collect(sort_by=None)["exp_name"])
    assert "cb1-99-da-xoa" not in names
    assert "CẢNH BÁO" in capsys.readouterr().out


# ── guard tập user ───────────────────────────────────────────────────────

def _write_peruser(rdir, exp_name, users, ndcg):
    np.savez(rdir / f"{exp_name}_peruser.npz",
             eval_users=np.asarray(users, dtype="int32"),
             **{"NDCG@10": np.asarray(ndcg, dtype="float32")})


def test_load_all_refuses_mismatched_user_sets(tmp_path):
    _write_peruser(tmp_path, "full_a", [0, 1, 2], [0.1, 0.2, 0.3])
    _write_peruser(tmp_path, "full_b", [0, 1, 9], [0.1, 0.2, 0.3])   # cùng độ dài, khác user
    with pytest.raises(SystemExit, match="tập user khác"):
        load_all("full_", "NDCG@10", results_dir=tmp_path)


def test_load_all_reports_missing_metric(tmp_path):
    _write_peruser(tmp_path, "full_a", [0, 1], [0.1, 0.2])
    with pytest.raises(SystemExit, match="NDCG@99"):
        load_all("full_", "NDCG@99", results_dir=tmp_path)


def test_bucket_table_refuses_mismatched_user_sets(tmp_path, monkeypatch):
    """Chốt lỗi M6: `sizes` lấy từ file đầu rồi áp cho file sau → bucket sai âm thầm."""
    _write_peruser(tmp_path, "full_a", [0, 1, 2], [0.1, 0.2, 0.3])
    _write_peruser(tmp_path, "full_b", [0, 1, 9], [0.1, 0.2, 0.3])
    monkeypatch.setattr("src.eval.user_bucket_analysis.train_history_size",
                        lambda users, split_dir=None: np.array([10, 50, 200]))
    with pytest.raises(SystemExit, match="tập eval_users khác"):
        bucket_table("full_", "NDCG@10", results_dir=tmp_path)


def test_bucket_table_dev_protocol_excludes_prefixed_results(tmp_path, monkeypatch):
    """Kết quả dev không có tiền tố, nên prefix="" sẽ vơ cả full_/coldstart_ vào."""
    _write_peruser(tmp_path, "cb1-01", [0, 1], [0.1, 0.2])
    _write_peruser(tmp_path, "full_cb1-01", [0, 1], [0.3, 0.4])
    _write_peruser(tmp_path, "coldstart_cb1-01", [0, 1], [0.5, 0.6])
    monkeypatch.setattr("src.eval.user_bucket_analysis.train_history_size",
                        lambda users, split_dir=None: np.array([10, 200]))

    dev_only = bucket_table("", "NDCG@10", results_dir=tmp_path, protocol="dev")
    assert list(dev_only["exp_name"]) == ["cb1-01"]
    assert len(bucket_table("", "NDCG@10", results_dir=tmp_path)) == 3


def test_bucket_table_rejects_unknown_protocol(tmp_path):
    with pytest.raises(ValueError, match="protocol"):
        bucket_table("", "NDCG@10", results_dir=tmp_path, protocol="bogus")


def test_bucket_table_assigns_users_to_correct_buckets(tmp_path, monkeypatch):
    _write_peruser(tmp_path, "full_a", [0, 1, 2, 3], [1.0, 2.0, 4.0, 8.0])
    monkeypatch.setattr("src.eval.user_bucket_analysis.train_history_size",
                        lambda users, split_dir=None: np.array([5, 30, 31, 101]))
    row = bucket_table("full_", "NDCG@10", results_dir=tmp_path).iloc[0]
    assert row["5–30"] == pytest.approx(1.5)      # user 0 và 1
    assert row["31–100"] == pytest.approx(4.0)    # user 2
    assert row[">100"] == pytest.approx(8.0)      # user 3
    assert (row["n_5–30"], row["n_31–100"], row["n_>100"]) == (2, 1, 1)


def test_paired_bucket_diffs_reports_per_bucket_difference(tmp_path, monkeypatch):
    users = list(range(8))
    _write_peruser(tmp_path, "ref", users, [0.1] * 8)
    _write_peruser(tmp_path, "alt", users, [0.3] * 4 + [0.0] * 4)
    monkeypatch.setattr("src.eval.user_bucket_analysis.train_history_size",
                        lambda users: np.array([10] * 4 + [200] * 4))
    df = paired_bucket_diffs("ref", ["alt"], results_dir=tmp_path).set_index("nhóm")
    assert df.loc["5–30", "chênh_lệch"] == pytest.approx(0.2)
    assert df.loc[">100", "chênh_lệch"] == pytest.approx(-0.1)
    # hiệu hằng số → CI suy biến về đúng điểm đó
    assert df.loc["5–30", "ci_low"] == pytest.approx(0.2)
    assert df.loc["5–30", "ci_high"] == pytest.approx(0.2)
    assert df.loc["5–30", "tỷ_lệ_tốt_hơn"] == pytest.approx(1.0)
    assert "31–100" not in df.index        # nhóm rỗng bị bỏ qua, không ném lỗi


def test_paired_bucket_diffs_refuses_mismatched_users(tmp_path, monkeypatch):
    _write_peruser(tmp_path, "ref", [0, 1], [0.1, 0.2])
    _write_peruser(tmp_path, "alt", [0, 9], [0.1, 0.2])
    monkeypatch.setattr("src.eval.user_bucket_analysis.train_history_size",
                        lambda users: np.array([10, 200]))
    with pytest.raises(SystemExit, match="tập eval_users khác"):
        paired_bucket_diffs("ref", ["alt"], results_dir=tmp_path)


def test_paired_bucket_diffs_p_value_is_float64_and_ties_counted(tmp_path, monkeypatch):
    """Chốt lỗi float32: p phải tính ở float64, và p = 1 khi mọi hiệu bằng 0."""
    users = list(range(40))
    ref = [0.0] * 20 + [0.5] * 20
    alt = [0.0] * 20 + [0.5 + 0.01 * (i + 1) for i in range(20)]   # 20 hoà, 20 tốt hơn
    _write_peruser(tmp_path, "ref", users, ref)
    _write_peruser(tmp_path, "alt", users, alt)
    _write_peruser(tmp_path, "same", users, ref)
    monkeypatch.setattr("src.eval.user_bucket_analysis.train_history_size",
                        lambda users: np.full(len(users), 10))
    df = paired_bucket_diffs("ref", ["alt", "same"], results_dir=tmp_path).set_index("exp_name")
    assert df.loc["alt", "tỷ_lệ_hoà"] == pytest.approx(0.5)
    assert 0.0 < df.loc["alt", "p_wilcoxon"] < 0.001
    assert isinstance(df.loc["alt", "p_wilcoxon"], float)
    assert df.loc["same", "p_wilcoxon"] == 1.0
