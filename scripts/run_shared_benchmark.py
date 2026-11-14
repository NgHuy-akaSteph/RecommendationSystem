"""Chạy bảng so sánh chung CF/CB: 9 mô hình × 3 giao thức.

Giao thức, theo đúng thứ tự phase 03:

    dev         20.000 user — chọn tham số, kiểm tra số liệu khớp bảng dev của CB
    full        158.774 user — bảng chính, chạy **một lần** sau khi khoá config
    coldstart   10% phim bị giữ lại — bảng cold-start item

Mỗi lần chạy bật `--save-topk` vì coverage/Gini/novelty cần danh sách gọi ý, và chạy
lại chỉ để lấy top-k thì quá đắt.

Tên kết quả có tiền tố giao thức (`full_`, `coldstart_`) nên kết quả dev không bị ghi
đè bởi kết quả full — cùng một cấu hình, ba giao thức, ba file.

Chạy:
    python scripts/run_shared_benchmark.py --protocol dev
    python scripts/run_shared_benchmark.py --protocol full
    python scripts/run_shared_benchmark.py --protocol coldstart
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLDSTART_DIR = ROOT / "data" / "processed" / "coldstart"
CHILD_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

# 9 mô hình của bảng chung. 5 config CF + 4 config CB, tất cả đã khoá tham số trên dev.
# Danh sách này phải khớp với `src.eval.benchmark_tables.CF_SPECS + CB_SPECS` — test
# `test_shared_benchmark_covers_all_table_specs` chặn trường hợp hai bên lệch nhau.
CONFIGS = [
    "baseline-random.yaml",
    "baseline-mostpop.yaml",
    "best-rs-itemcf-binary-pos.yaml",
    "best-cf-als.yaml",
    "best-rs-bpr.yaml",
    "best-cb1-tfidf.yaml",
    "best-cb2-bm25.yaml",
    "best-cb3-genome.yaml",
    "best-cb4-hybrid-content.yaml",
]

# Sweep ALS (reg × iterations) chạy riêng trên dev để cân bằng số config CF với CB.
ALS_SWEEP = [
    "cf-als-f064-a005-reg0p001.yaml",
    "cf-als-f064-a005-reg0p1.yaml",
    "cf-als-f064-a005-reg1p0.yaml",
    "cf-als-f064-a005-it030.yaml",
    "cf-als-f064-a005-reg0p001-it030.yaml",
    "cf-als-f064-a005-reg0p1-it030.yaml",
    "cf-als-f064-a005-reg1p0-it030.yaml",
]


def result_name(config: Path, users: str, split_dir: Path | None) -> str:
    """Khớp với src.eval.run_experiment_from_config.experiment_name."""
    import yaml
    base = yaml.safe_load(config.read_text(encoding="utf-8"))["exp_name"]
    prefix = (f"{split_dir.name}_" if split_dir else "") + ("full_" if users == "full" else "")
    return prefix + base


def run_one(config_name: str, users: str, split_dir: Path | None,
            force: bool) -> str | None:
    config = ROOT / "configs" / config_name
    if not config.exists():
        print(f"THIẾU CONFIG: {config.name}")
        return config_name

    exp = result_name(config, users, split_dir)
    metrics = ROOT / "results" / f"{exp}_metrics.csv"
    topk = ROOT / "results" / f"{exp}_topk.npy"
    if metrics.exists() and topk.exists() and not force:
        print(f"SKIP (đã có): {exp}")
        return None

    cmd = [sys.executable, "-m", "src.eval.run_experiment_from_config", str(config),
           "--users", users, "--save-topk"]
    if split_dir is not None:
        cmd += ["--split-dir", str(split_dir)]
    print(f"\n--- {exp} ---")
    if subprocess.run(cmd, cwd=str(ROOT), env=CHILD_ENV).returncode != 0:
        print(f"FAILED: {exp}")
        return config_name
    return None


def main(protocol: str = "dev", force: bool = False, als_sweep: bool = False) -> None:
    if protocol == "dev":
        users, split_dir = "dev", None
    elif protocol == "full":
        users, split_dir = "full", None
    elif protocol == "coldstart":
        users, split_dir = "dev", COLDSTART_DIR
    else:
        raise SystemExit(f"protocol lạ: {protocol}")

    if protocol == "coldstart" and not (COLDSTART_DIR / "train.parquet").exists():
        raise SystemExit(
            "Thiếu split cold-start. Chạy: python -m src.eval.cold_start_item_split")

    configs = list(CONFIGS)
    if als_sweep:
        # Sweep chỉ chạy trên dev: tham số phải chọn trên dev, `full` chạy đúng một lần
        # với config đã khoá.
        if protocol != "dev":
            raise SystemExit("Sweep ALS chỉ chạy trên dev (tham số không được chọn trên full).")
        configs += ALS_SWEEP

    print(f"\n{'=' * 70}\n{len(configs)} config | protocol={protocol} "
          f"users={users}{' split=' + split_dir.name if split_dir else ''}\n{'=' * 70}")

    failed = []
    for config_name in configs:
        bad = run_one(config_name, users, split_dir, force)
        if bad:
            failed.append(bad)

    if failed:
        print("\nThí nghiệm lỗi: " + ", ".join(failed))
        sys.exit(1)
    print("\nXong toàn bộ.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="dev", choices=["dev", "full", "coldstart"])
    parser.add_argument("--force", action="store_true", help="Chạy lại cả khi đã có kết quả")
    parser.add_argument("--als-sweep", action="store_true",
                        help="Thêm lưới ALS reg × iterations (chỉ hợp lệ với --protocol dev)")
    main(**vars(parser.parse_args()))
