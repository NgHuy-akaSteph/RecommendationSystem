"""Sinh file YAML cho mọi thí nghiệm từ lưới khai báo ở ablation_grid_definitions.

File YAML là **sản phẩm sinh ra**, không phải đầu vào: thêm một giá trị vào lưới là sửa
đúng một dòng rồi chạy lại script này (hoặc chạy `run_all_experiments.py`, nó tự gọi).

Chạy: python scripts/generate_ablation_configs.py [--clean]
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation_grid_definitions import CONFIG_DIR, ROOT, all_configs  # noqa: E402,F401

# best-* do select_best_configs.py sinh ra; phải nằm trong danh sách dọn, nếu không file
# cũ sẽ sống sót qua --clean và vẫn được collect() coi là config hợp lệ
CLEAN_PREFIXES = ("cb1-", "cb2-", "cb3-", "cb4-", "cf-", "best-")


def main(clean: bool = False) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if clean:
        for prefix in CLEAN_PREFIXES:
            for old in CONFIG_DIR.glob(f"{prefix}*.yaml"):
                old.unlink()
                print(f"  xoá {old.name}")

    configs = all_configs()
    for exp_name, model, params in configs:
        doc = {
            "exp_name": exp_name, "model": model, "params": params,
            "users": "dev", "k_list": [5, 10, 20, 50], "save_topk": False,
        }
        banner = "# Sinh bởi scripts/generate_ablation_configs.py — không sửa tay\n"
        (CONFIG_DIR / f"{exp_name}.yaml").write_text(
            banner + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    print(f"Đã sinh {len(configs)} config trong {CONFIG_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true",
                        help="Xoá config cb*/cf-*/best-* cũ trước khi sinh")
    main(**vars(parser.parse_args()))
