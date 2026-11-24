"""Sinh bảng kiểm chứng demo cold-start — phục vụ §6.6.4 của báo cáo.

§6.6.4 trình bày "bằng chứng trực quan" cho luận điểm trung tâm của đồ án: với
một phim cold-start, CF **không xếp hạng được** (điểm bằng 0, rơi xuống đáy bảng)
còn CB **vẫn xếp hạng được** (vì dựng hồ sơ từ nội dung phim).

Các con số đó trước đây chỉ tồn tại trong màn hình demo — không có bảng CSV nào
lưu lại, nên `check_report_numbers.py` không kiểm chứng được. Script này chạy
đúng phép đo mà demo chạy (`RecommenderService.coldstart_movies_for`) và ghi kết
quả ra CSV để checker đối chiếu.

Output:
    results/final/demo-coldstart-comparison.csv

Chạy:
    venv/bin/python -m src.eval.build_demo_coldstart_comparison
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import paths

# Artifact demo — phải sinh trước bằng:
#   venv/bin/python scripts/save_demo_artifacts.py --split coldstart
ARTIFACTS_DIR = paths.ROOT / "demo" / "artifacts"
MODEL_KEYS = ["cf-als", "cb2-bm25"]

# Phim cold-start dùng trong ảnh chụp màn hình §6.6.4 (*Nixon*, 1995).
NIXON_ITEM_IDX = 13


def build_comparison() -> pd.DataFrame:
    """Chạy phép so sánh CF vs CB trên một phim cold-start.

    Returns:
        DataFrame với cột `model`, `cold_item_score`, `cold_item_rank`,
        `user_id`, `n_items`.
    """
    # Import cục bộ — `src.api` kéo theo phụ thuộc demo (streamlit) chỉ khi cần.
    from src.api.recommender_service import RecommenderService

    service = RecommenderService(
        artifacts_dir=ARTIFACTS_DIR,
        model_keys=MODEL_KEYS,
        processed_dir=paths.PROCESSED_DIR,
    )
    result = service.coldstart_movies_for(NIXON_ITEM_IDX, n=10)
    if not result.get("available"):
        raise SystemExit(
            "Chưa có artifact cold-start. Chạy: "
            "venv/bin/python scripts/save_demo_artifacts.py --split coldstart"
        )

    rows = []
    for key, payload in result["models"].items():
        rows.append({
            "model": key,
            "cold_item_score": float(payload["cold_item_score"]),
            "cold_item_rank": payload["cold_item_rank"],
            "user_id": payload["user_id"],
            "n_items": payload["count"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = build_comparison()
    out = paths.FINAL_DIR / "demo-coldstart-comparison.csv"
    df.to_csv(out, index=False)
    print(f"Đã ghi {out}")
    for r in df.itertuples(index=False):
        print(f"  {r.model}: score={r.cold_item_score:.6f} rank={r.cold_item_rank} "
              f"user={r.user_id}")


if __name__ == "__main__":
    main()
