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


# ── runner và bảng phải phủ đúng nhau ───────────────────────────────────

def test_shared_benchmark_covers_all_table_specs():
    """`run_shared_benchmark.CONFIGS` phải phủ hết config của bảng benchmark.

    Lỗi đã xảy ra thật: runner chỉ có 8 config (thiếu `best-cb4-hybrid-content.yaml`)
    trong khi `CB_SPECS` đã có CB-4 → chạy lại runner sẽ không sinh dòng CB-4 và
    `benchmark_tables` fail với "Thiếu kết quả".
    """
    import sys
    sys.path.insert(0, str(paths.ROOT / "scripts"))
    import run_shared_benchmark as runner

    runner_configs = set(runner.CONFIGS)
    table_configs = {spec.config for spec in ALL_SPECS}
    missing = table_configs - runner_configs
    assert not missing, f"runner thiếu config của bảng benchmark: {sorted(missing)}"
    # Runner không được có config thừa mà bảng không dùng — config thừa chạy tốn thời gian
    # và tạo kết quả mồ côi.
    extra = runner_configs - table_configs
    assert not extra, f"runner có config không dùng trong bảng: {sorted(extra)}"


# ── bằng chứng protocol phải có trong mọi bảng báo cáo ──────────────────

@pytest.mark.parametrize("filename", [
    "benchmark-cf.csv", "benchmark-cb.csv", "benchmark-coldstart.csv",
    "dev-summary.csv", "full-summary.csv", "coldstart-item.csv",
])
def test_final_tables_have_protocol_evidence(filename):
    """Mọi dòng bảng báo cáo phải mang `split_id`/`catalog_id` khác rỗng.

    Đã xảy ra thật: `dev-summary.csv` thiếu cột này ở dòng CB-4 vì file metrics
    được sinh trước khi `run_and_save` bắt đầu ghi `split_id`. Bảng `full` và
    `coldstart` thì có → bảng dev trông như thiếu bằng chứng protocol dù nó có.
    """
    path = paths.FINAL_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} chưa được sinh — chạy build_final_tables.py trước")
    df = pd.read_csv(path)
    for col in ("split_id", "catalog_id"):
        assert col in df.columns, f"{filename}: thiếu cột {col}"
        assert df[col].notna().all(), f"{filename}: có dòng thiếu {col}"
        assert (df[col].astype(str).str.strip() != "").all(), \
            f"{filename}: có dòng {col} rỗng"
    # Mọi dòng trong một bảng phải cùng split — nếu lệch thì "cùng giao thức" là lời nói dối
    assert df["split_id"].nunique() == 1, \
        f"{filename}: các dòng thuộc khác split: {sorted(df['split_id'].unique())}"
    assert df["catalog_id"].nunique() == 1, \
        f"{filename}: các dòng thuộc khác catalog: {sorted(df['catalog_id'].unique())}"


@pytest.mark.parametrize("group", ["cf", "cb1", "cb2", "cb3", "cb4"])
def test_ablation_tables_have_protocol_evidence(group):
    """Bảng sweep trong `results/final/` phải đủ bằng chứng protocol từng dòng.

    Đã xảy ra thật: 7 config chạy trước khi thêm cột `split_id` khiến bảng ablation
    thiếu bằng chứng ở đúng những dòng đó — luận điểm "đã quét đủ sweep" mà bảng
    không đủ giấy tờ thì không kiểm chứng được từ repo.
    """
    path = paths.FINAL_DIR / f"ablation-{group}.csv"
    if not path.exists():
        pytest.skip(f"{path.name} chưa được sinh — chạy collect_results_to_table --ablation")
    df = pd.read_csv(path)
    assert len(df) > 0, f"{path.name}: rỗng"
    for col in ("split_id", "catalog_id"):
        assert col in df.columns, f"{path.name}: thiếu cột {col}"
        assert df[col].notna().all(), \
            f"{path.name}: {df.loc[df[col].isna(), 'exp_name'].tolist()} thiếu {col}"
        assert (df[col].astype(str).str.strip() != "").all(), \
            f"{path.name}: có dòng {col} rỗng"
    # Mỗi config chỉ xuất hiện đúng một lần — trùng là ghép nhầm bản kết quả
    assert df["exp_name"].is_unique, \
        f"{path.name}: có exp_name trùng: {df['exp_name'][df['exp_name'].duplicated()].tolist()}"


@pytest.mark.parametrize("group, prefixes", [
    ("cf", ("cf-", "rs-")), ("cb1", ("cb1-",)), ("cb2", ("cb2-",)),
    ("cb3", ("cb3-",)), ("cb4", ("cb4-",)),
])
def test_ablation_covers_every_config_of_its_group(group, prefixes):
    """Bảng ablation phải phủ **đúng** mọi config thuộc nhóm — không thiếu, không thừa.

    Đã xảy ra thật: bảng CF trước đây chỉ lấy tiền tố `cf-` (18 dòng) trong khi
    sweep CF gồm cả họ `rs-*` (Item-CF, BPR) — 51 config. Tài liệu tuyên bố "CF đã
    quét 51 config" nhưng bằng chứng commit chỉ có 18 dòng: tuyên bố không kiểm
    chứng được từ repo.
    """
    path = paths.FINAL_DIR / f"ablation-{group}.csv"
    if not path.exists():
        pytest.skip(f"{path.name} chưa được sinh — chạy collect_results_to_table --ablation")
    df = pd.read_csv(path)

    in_configs = {p.stem for p in paths.CONFIG_DIR.glob("*.yaml")
                  if p.stem.startswith(prefixes)}
    in_table = set(df["exp_name"])

    missing = sorted(in_configs - in_table)
    assert not missing, \
        f"{path.name}: thiếu {len(missing)} config đã sweep: {missing[:5]}"

    # Config thừa trong bảng là kết quả mồ côi — config không còn trong configs/
    extra = sorted(in_table - in_configs)
    assert not extra, \
        f"{path.name}: có {len(extra)} config không còn trong configs/: {extra[:5]}"
