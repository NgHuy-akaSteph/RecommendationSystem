"""Chạy một thí nghiệm mô tả bằng file YAML.

Mọi thí nghiệm đều đi qua đúng một đường dẫn code này, nên các thuật toán được so
sánh trên cùng dữ liệu, cùng giao thức, cùng định nghĩa metric.

Ví dụ config:
    exp_name: baseline-mostpop
    model: mostpop
    params: {}
    users: dev          # dev (20k user) hoặc full
    k_list: [5, 10, 20, 50]
    save_topk: false

Chạy:
    python -m src.eval.run_experiment_from_config configs/baseline-mostpop.yaml
    python -m src.eval.run_experiment_from_config configs/best-cf-als.yaml --users full --save-topk
    python -m src.eval.run_experiment_from_config configs/best-cf-als.yaml --split-dir data/processed/coldstart
"""

import argparse
from pathlib import Path

import yaml

from src.data.build_interaction_matrices import load_dataset, load_item_features
from src.data.catalog_identity import catalog_id, split_id
from src.eval import topk_ranking_metrics as metrics_mod
from src.eval.batched_scoring_evaluator import run_and_save
from src.eval.model_registry import build_model


def experiment_name(base: str, users: str, split_dir: Path | None) -> str:
    """Tiền tố cho biết thí nghiệm chạy ở giao thức nào — tránh ghi đè kết quả dev."""
    prefix = ""
    if split_dir is not None:
        prefix += f"{split_dir.name}_"
    if users == "full":
        prefix += "full_"
    return prefix + base


def run_config(
    config_path: Path,
    users_override: str | None = None,
    split_dir: Path | None = None,
    save_topk_override: bool = False,
) -> dict[str, float]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    users = users_override or config.get("users", "dev")
    exp_name = experiment_name(config["exp_name"], users, split_dir)
    k_list = tuple(config.get("k_list", metrics_mod.DEFAULT_K_LIST))
    save_topk = save_topk_override or config.get("save_topk", False)

    print(f"[{exp_name}] model={config['model']} users={users} params={config.get('params', {})}")
    dataset = load_dataset(users=users, split_dir=split_dir)
    item_features = load_item_features()
    print(f"  {len(dataset.eval_users):,} user đánh giá, {dataset.n_items:,} item ứng viên")

    model = build_model(config["model"], config.get("params"))
    summary = run_and_save(
        model,
        dataset,
        exp_name=exp_name,
        item_features=item_features,
        k_list=k_list,
        save_topk=save_topk,
        # Ghi cả split vào metadata: nếu không, dấu vết duy nhất cho biết một dòng kết quả
        # thuộc giao thức nào là tiền tố trong tên file — đổi tên file là mất giao thức.
        config={**config, "users": users, "split": split_dir.name if split_dir else "",
                "split_id": split_id(split_dir.name if split_dir else None),
                "catalog_id": catalog_id()},
    )

    print(metrics_mod.format_table(summary, k_list))
    print(f"  fit {summary['fit_s']}s | score {summary['score_s']}s")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Đường dẫn file YAML cấu hình")
    parser.add_argument(
        "--users", choices=["dev", "full"], default=None, help="Ghi đè trường users của config"
    )
    parser.add_argument(
        "--split-dir", type=Path, default=None,
        help="Thư mục chứa train/test thay thế (ví dụ split cold-start item)",
    )
    parser.add_argument(
        "--save-topk", action="store_true",
        help="Lưu {exp}_topk.npy — bắt buộc cho coverage/novelty/ILD ở phase 4",
    )
    args = parser.parse_args()
    run_config(args.config, args.users, args.split_dir, args.save_topk)


if __name__ == "__main__":
    main()
