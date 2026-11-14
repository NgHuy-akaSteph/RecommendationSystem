"""Chạy các đánh giá cuối của phase 4 từ configs/best-*.yaml.

Hai giao thức:
    --full       toàn bộ 158.774 user hợp lệ (dev chỉ là mẫu 20k để tinh chỉnh)
    --coldstart  split giữ lại 10% phim, dùng để đo đúng luận điểm chính của đề tài

Cả hai đều bật `--save-topk` vì coverage / novelty / ILD cần danh sách gợi ý, và chạy lại
chỉ để lấy top-k thì quá đắt.

Chạy: python scripts/run_final_evaluations.py --full --coldstart [--force]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLDSTART_DIR = ROOT / "data" / "processed" / "coldstart"
CHILD_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def best_configs() -> list[Path]:
    """configs/best-*.yaml do select_best_configs.py sinh ra, một file mỗi thuật toán."""
    return sorted((ROOT / "configs").glob("best-*.yaml"))


def result_name(config: Path, users: str, split_dir: Path | None) -> str:
    """Khớp với src.eval.run_experiment_from_config.experiment_name."""
    import yaml
    base = yaml.safe_load(config.read_text(encoding="utf-8"))["exp_name"]
    prefix = (f"{split_dir.name}_" if split_dir else "") + ("full_" if users == "full" else "")
    return prefix + base


def run_protocol(users: str, split_dir: Path | None, force: bool) -> list[str]:
    configs = best_configs()
    label = f"users={users}" + (f" split={split_dir.name}" if split_dir else "")
    print(f"\n{'=' * 70}\n{len(configs)} thuật toán | {label}\n{'=' * 70}")

    failed = []
    for config in configs:
        exp = result_name(config, users, split_dir)
        # Kiểm tra cả topk: run_and_save ghi metrics TRƯỚC topk, nên một lần chạy bị ngắt
        # giữa hai bước sẽ bị skip vĩnh viễn và beyond_accuracy sau đó nổ vì thiếu topk.
        done = ((ROOT / "results" / f"{exp}_metrics.csv").exists()
                and (ROOT / "results" / f"{exp}_topk.npy").exists())
        if done and not force:
            print(f"SKIP (đã có): {exp}")
            continue

        cmd = [sys.executable, "-m", "src.eval.run_experiment_from_config", str(config),
               "--users", users, "--save-topk"]
        if split_dir is not None:
            cmd += ["--split-dir", str(split_dir)]
        print(f"\n--- {exp} ---")
        if subprocess.run(cmd, cwd=str(ROOT), env=CHILD_ENV).returncode != 0:
            print(f"FAILED: {exp}")
            failed.append(exp)
    return failed


def main(full: bool = False, coldstart: bool = False, force: bool = False) -> None:
    if not (full or coldstart):
        raise SystemExit("Chọn ít nhất một trong --full / --coldstart")

    failed = []
    if full:
        failed += run_protocol("full", None, force)
    if coldstart:
        if not (COLDSTART_DIR / "train.parquet").exists():
            raise SystemExit("Thiếu split cold-start. Chạy: python -m src.eval.cold_start_item_split")
        # Cold-start chấm trên tập dev: user hợp lệ là những người có phim held-out
        # trong test, load_dataset tự lọc bằng has_relevant nên không cần danh sách riêng.
        failed += run_protocol("dev", COLDSTART_DIR, force)

    if failed:
        print("\nThí nghiệm lỗi: " + ", ".join(failed))
        sys.exit(1)
    print("\nXong toàn bộ.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Chạy trên toàn bộ user")
    parser.add_argument("--coldstart", action="store_true", help="Chạy trên split cold-start item")
    parser.add_argument("--force", action="store_true", help="Chạy lại cả khi đã có kết quả")
    main(**vars(parser.parse_args()))
