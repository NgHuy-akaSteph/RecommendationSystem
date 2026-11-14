"""Chạy toàn bộ thí nghiệm đã khai báo trong grid, rồi sinh bảng ablation.

Danh sách config lấy trực tiếp từ generate_ablation_configs (một nguồn sự thật),
nên thêm một giá trị vào sweep là script này tự chạy thêm.

Chạy: python scripts/run_all_experiments.py [--force] [--only cf-]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # chạy được cả `python -m scripts.…`
from generate_ablation_configs import ROOT, all_configs, main as generate_configs

# Console Windows mặc định cp1252 khi output bị pipe → print tiếng Việt ném
# UnicodeEncodeError và giết tiến trình con giữa chừng. Ép UTF-8 cho mọi subprocess.
CHILD_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def _params_changed(config_path, result_file) -> bool:
    """True nếu params trong config khác params đã sinh ra kết quả đã lưu."""
    import pandas as pd
    import yaml
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        stored = pd.read_csv(result_file)["params"].iloc[0]
    except (KeyError, IndexError, OSError, ValueError):
        return True   # không đọc được thì chạy lại cho chắc
    if json.dumps(cfg.get("params", {}), ensure_ascii=False) == stored:
        return False
    print(f"RERUN (params da doi): {config_path.stem}")
    return True


def main(force: bool = False, only: str | None = None) -> None:
    # Config là sản phẩm sinh ra, không phải đầu vào: sinh lại trước mỗi lần chạy thì
    # thêm một điểm vào grid là chạy được ngay, không âm thầm bỏ qua vì thiếu file YAML.
    generate_configs()
    configs = [(name, ROOT / "configs" / f"{name}.yaml") for name, _, _ in all_configs()]
    if only:
        configs = [(n, p) for n, p in configs if n.startswith(only)]

    done, skipped, failed, missing = 0, 0, [], []
    for exp_name, path in configs:
        if not path.exists():
            print(f"THIẾU CONFIG: {path.name}")
            missing.append(exp_name)
            continue

        # Không skip mù theo sự tồn tại của file kết quả: nhiều config phase 3 nhúng
        # tham số của mô hình con (BEST_CB2...) mà tên thí nghiệm không mã hoá, nên sửa
        # tham số đó xong vẫn trùng tên file và kết quả cũ bị giữ lại một cách âm thầm.
        # So thẳng params đã lưu trong CSV với params của config hiện tại.
        result_file = ROOT / "results" / f"{exp_name}_metrics.csv"
        if result_file.exists() and not force and not _params_changed(path, result_file):
            print(f"SKIP (đã có kết quả, dùng --force để chạy lại): {exp_name}")
            skipped += 1
            continue

        print(f"\n{'=' * 60}\nRUNNING: {exp_name}\n{'=' * 60}")
        proc = subprocess.run(
            [sys.executable, "-m", "src.eval.run_experiment_from_config", str(path)],
            cwd=str(ROOT), env=CHILD_ENV,
        )
        if proc.returncode != 0:
            print(f"FAILED: {exp_name}")
            failed.append(exp_name)
        else:
            done += 1

    print(f"\n{'=' * 60}\nCOLLECTING RESULTS  (chạy {done}, bỏ qua {skipped}, lỗi {len(failed)})\n{'=' * 60}")
    if failed:
        print("Thí nghiệm lỗi: " + ", ".join(failed))
    if missing:
        print("Thiếu config: " + ", ".join(missing))
    subprocess.run(
        [sys.executable, "-m", "src.eval.collect_results_to_table",
         "--ablation", "--output", "results/summary-phase2.csv"],
        cwd=str(ROOT), env=CHILD_ENV,
    )
    # Thiếu config cũng là lỗi: bảng ablation thiếu dòng mà exit 0 là đúng kiểu "số sai
    # mà không có gì báo sai" — chính lớp lỗi mà thay đổi này đang sửa.
    sys.exit(1 if (failed or missing) else 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Chạy lại cả khi đã có kết quả")
    parser.add_argument("--only", default=None, help="Chỉ chạy exp_name bắt đầu bằng chuỗi này")
    main(**vars(parser.parse_args()))
