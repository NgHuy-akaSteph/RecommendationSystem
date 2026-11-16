"""Sinh artifact cho demo: fit một lần rồi lưu xuống đĩa.

Vì sao cần script này: `fit` trên 158.774 user mất ~15s (ALS) và `load_dataset`
mất ~18s. Demo phải mở trong vài giây, nên fit **một lần** ra artifact rồi demo chỉ nạp.

Chạy:
    venv/bin/python scripts/save_demo_artifacts.py

Mặc định sinh 2 mô hình: `cf-als` (mạnh nhất nhóm CF) và `cb2-bm25` (mạnh nhất nhóm
CB, đồng thời dùng được cho gợi ý "phim tương tự" và giải thích bằng tag).

Tuỳ chọn:
    --models cf-als cb2-bm25 cb4-hybrid
    --split full            # hoặc coldstart
    --out demo/artifacts
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import paths
from src.api.model_artifacts import save_artifact, save_train_csr
from src.data.build_interaction_matrices import load_dataset, load_item_features
from src.eval.model_registry import build_model

# Config khoá tham số — đúng file trong `configs/` nên không chép tay tham số ở đây.
DEFAULT_CONFIGS = {
    "cf-als": "best-cf-als.yaml",
    "cb2-bm25": "best-cb2-bm25.yaml",
    "cb4-hybrid": "best-cb4-hybrid-content.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", default=["cf-als", "cb2-bm25"],
        help="Khoá mô hình cần sinh artifact (xem DEFAULT_CONFIGS)",
    )
    parser.add_argument("--out", type=Path, default=paths.ROOT / "demo" / "artifacts")
    parser.add_argument("--split", choices=["full", "coldstart"], default="full")
    args = parser.parse_args()

    unknown = [m for m in args.models if m not in DEFAULT_CONFIGS]
    if unknown:
        raise SystemExit(f"Mô hình chưa có config khoá: {unknown}. Có: {sorted(DEFAULT_CONFIGS)}")

    split_dir = None
    out_dir = args.out
    if args.split == "coldstart":
        split_dir = paths.PROCESSED_DIR / "coldstart"
        # Artifact cold-start nằm ở thư mục riêng: nó phải fit trên split
        # cold-start (không còn tương tác của 1.381 phim held-out), nên **không**
        # được trộn với artifact của split `full`.
        out_dir = args.out / "coldstart"

    t0 = time.time()
    dataset = load_dataset(users="full", split_dir=split_dir)
    item_features = load_item_features()
    print(f"[{args.split}] nạp dữ liệu xong: {len(dataset.eval_users):,} user đánh giá "
          f"({time.time() - t0:.1f}s)")

    for model_key in args.models:
        config_path = paths.CONFIG_DIR / DEFAULT_CONFIGS[model_key]
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        params = config.get("params", {})

        t1 = time.time()
        model = build_model(model_key, params)
        model.fit(dataset.train, item_features)
        fit_s = time.time() - t1

        target = save_artifact(
            model_key, model, out_dir,
            manifest_extra={
                "exp_name": config.get("exp_name", ""),
                "config": config_path.name,
                "params": params,
                "split": args.split,
                "n_users": int(dataset.train.n_users),
                "n_items": int(dataset.train.n_items),
                "fit_s": round(fit_s, 2),
            },
        )
        size_mb = sum(f.stat().st_size for f in target.rglob("*")) / 1024 / 1024
        print(f"  [{model_key}] fit {fit_s:.1f}s -> {target} ({size_mb:.1f} MB)")

    if args.split == "full":
        save_train_csr(dataset.train, args.out)
        print(f"  [train_csr] -> {args.out / 'train_csr.npz'}")
    else:
        # Cold-start cần ma trận train *đã xoá* tương tác của phim held-out để mask
        # lịch sử đúng.
        save_train_csr(dataset.train, out_dir)
        print(f"  [train_csr] -> {out_dir / 'train_csr.npz'}")

    print(f"Xong ({time.time() - t0:.1f}s). Chạy demo: venv/bin/streamlit run demo/app.py")


if __name__ == "__main__":
    main()
