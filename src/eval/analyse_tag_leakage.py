"""Chẩn đoán leakage thời gian của đặc trưng tag (CB-1) trên split temporal per-user.

Bối cảnh: split là temporal **per-user** (80% cũ / 20% mới của từng user), nhưng mọi
đặc trưng phía item lại tính trên toàn corpus. Tag của MovieLens chỉ bắt đầu ~2006 nên
với user kết thúc lịch sử train trước đó, toàn bộ tag mà CB-1 dùng đều được tạo *sau*
thời điểm dự đoán. Cắt tag theo timestamp không cứu được: ở phân vị 25 của mốc train
chỉ còn 0 tag, tức CB-1 "sạch leakage" là CB-1 rỗng.

Vì vậy không đo "bỏ leakage rồi còn bao nhiêu điểm" mà đo thứ trả lời đúng câu hỏi của
báo cáo: **thứ hạng giữa các phương pháp có ổn định qua các thời kỳ user không?** Nếu
CB-1 dẫn CB-3 đều tay ở mọi tầng thì kết luận của đồ án bền, bất kể mức lạc quan tuyệt
đối của con số.

Metric per-user đã lưu ở results/*_peruser.npz nên không cần chạy lại mô hình nào.

Chạy: python -m src.eval.analyse_tag_leakage
"""

import numpy as np
import pandas as pd

from src import paths

# Mốc chia tầng theo thời điểm kết thúc lịch sử train của user, so với "kỷ nguyên tag".
# Tag đầu tiên của MovieLens xuất hiện 2005-12; 2012 là mốc tag đã đủ dày.
ERA_BOUNDS = [("Trước kỷ nguyên tag (< 2006)", None, "2006-01-01"),
              ("Giao thời (2006–2012)", "2006-01-01", "2012-01-01"),
              ("Tag đã dày (>= 2012)", "2012-01-01", None)]

MODELS = {
    # (tên config gốc, nhãn hiển thị). Tên thực tế của file kết quả có tiền tố giao
    # thức ("full_", "coldstart_") — cùng quy ước với `experiment_name`.
    "cb4-03-timedecay-hl0365": "CB-1 TF-IDF (tag)",
    "cb3-04d-item-knn-tau020-k100": "CB-3 Genome",
    "cb1-00-genres-binary-profile": "Chỉ genres (không leak)",
    "baseline-mostpop": "MostPop",
    # CF cũng chịu lệch thời gian ở mức giao thức: ma trận train chứa rating của user khác
    # nằm sau mốc train của user đang xét. Đưa vào cùng bảng để chương giới hạn có số liệu
    # cho cả 8 thuật toán chứ không chỉ lập luận suông.
    "rs-itemcf-k200-binary-pos": "Item-CF (RS)",
    "cf-als-f064-a005-reg0p1": "ALS",
    "rs-bpr-f64-i10-reg0p00001-lr0p01": "BPR (RS)",
    "cb2-01-k1-05-b000": "CB-2 BM25",
}

# Mô hình nào có sẵn ở giao thức nào. `cb1-00-genres-binary-profile` chỉ chạy trên
# dev và full (nó là mốc "không leak" nên cần ở cả hai để so sánh tầng thời kỳ).
PROTOCOL_PREFIX = {"dev": "", "full": "full_"}


def user_train_end() -> pd.Series:
    """Timestamp của tương tác train cuối cùng mỗi user = thời điểm dự đoán của user đó."""
    train = pd.read_parquet(paths.TRAIN_PARQUET, columns=["user_idx", "ts"])
    return train.groupby("user_idx")["ts"].max()


def load_per_user(exp_name: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    path = paths.RESULTS_DIR / f"{exp_name}_peruser.npz"
    if not path.exists():
        raise SystemExit(f"Thiếu {path.name}. Chạy thí nghiệm '{exp_name}' trước: "
                         f"python -m src.eval.run_experiment_from_config configs/{exp_name}.yaml")
    data = np.load(path)
    return data["eval_users"], data[metric]


def stratify(metric: str = "NDCG@10", protocol: str = "full") -> pd.DataFrame:
    """Trung bình metric của từng mô hình trên từng tầng thời kỳ user."""
    train_end = user_train_end()
    prefix = PROTOCOL_PREFIX[protocol]
    rows = []
    ref_users = None
    for exp_name, label in MODELS.items():
        file_name = prefix + exp_name
        users, values = load_per_user(file_name, metric)
        # Nếu hai mô hình chấm trên tập user khác nhau thì bảng tỷ lệ bên dưới đang so
        # hai quần thể khác nhau mà không có dấu hiệu gì nhìn thấy được
        if ref_users is None:
            ref_users = users
        elif not np.array_equal(users, ref_users):
            raise SystemExit(f"'{file_name}' dùng tập eval_users khác các mô hình còn lại")
        ts = train_end.reindex(users).to_numpy()
        row = {"Mô hình": label}
        for era_label, lo, hi in ERA_BOUNDS:
            mask = np.ones(len(users), dtype=bool)
            if lo is not None:
                mask &= ts >= pd.Timestamp(lo).timestamp()
            if hi is not None:
                mask &= ts < pd.Timestamp(hi).timestamp()
            row[era_label] = float(values[mask].mean()) if mask.any() else np.nan
            row[f"n_{era_label}"] = int(mask.sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("Mô hình")


def tag_volume_vs_popularity() -> dict[str, float]:
    """Số tag mỗi phim có tương quan với độ phổ biến không?

    Nếu có, TF-IDF vô tình mã hoá độ phổ biến tương lai: phim được tag nhiều cho vector
    dày hơn nên cosine cao hơn với nhiều profile, kể cả sau khi L2-norm.
    """
    from src.data.build_interaction_matrices import load_dataset
    tag_docs = pd.read_parquet(paths.ITEM_TAG_DOCS_PARQUET)
    pop = load_dataset("dev").train.item_popularity
    n_tags = tag_docs.sort_values("item_idx")["n_tags"].to_numpy()
    return {
        "pearson_ntags_pop": float(np.corrcoef(n_tags, pop)[0, 1]),
        "spearman_ntags_pop": float(pd.Series(n_tags).corr(pd.Series(pop), method="spearman")),
        "items_khong_tag": int((n_tags == 0).sum()),
    }


def leakage_summary() -> pd.DataFrame:
    """Các con số nền của lập luận leakage, để báo cáo trích từ CSV thay vì từ console."""
    first_tag = pd.read_csv(paths.RAW_TAGS, usecols=["timestamp"])["timestamp"].min()
    dev_end = user_train_end().reindex(np.load(paths.DEV_USERS_NPY))
    stats = {
        "Ngày gán tag sớm nhất (toàn bộ tags.csv)": pd.to_datetime(first_tag, unit="s").date().isoformat(),
        "Trung vị ngày kết thúc huấn luyện (user tập phát triển)":
            pd.to_datetime(dev_end.median(), unit="s").date().isoformat(),
    }
    volume = tag_volume_vs_popularity()
    stats["Pearson(số tag của phim, độ phổ biến)"] = f"{volume['pearson_ntags_pop']:.3f}"
    stats["Spearman(số tag của phim, độ phổ biến)"] = f"{volume['spearman_ntags_pop']:.3f}"
    stats["Số phim không có tag"] = f"{volume['items_khong_tag']}"
    return pd.DataFrame({"Chỉ số": list(stats), "Giá trị": list(stats.values())})


def main(protocol: str = "full") -> None:
    print(f"=== Phân tầng theo thời kỳ kết thúc train của user (NDCG@10) "
          f"[protocol={protocol}] ===\n")
    table = stratify("NDCG@10", protocol)
    final_dir = paths.RESULTS_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if protocol == "full" else f"-{protocol}"
    table.to_csv(final_dir / f"tag-leakage-strata{suffix}.csv")
    leakage_summary().to_csv(final_dir / f"tag-leakage-summary{suffix}.csv", index=False)
    era_cols = [e[0] for e in ERA_BOUNDS]
    print(table[era_cols].to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nSố user mỗi tầng:",
          {e: int(table[f"n_{e}"].iloc[0]) for e in era_cols})

    print("\n=== Tỷ lệ dẫn trước của CB-1 so với các mốc (cùng tầng) ===")
    cb1 = table.loc["CB-1 TF-IDF (tag)", era_cols]
    for other in ["CB-3 Genome", "MostPop", "Chỉ genres (không leak)",
                  "Item-CF (RS)", "ALS", "BPR (RS)", "CB-2 BM25"]:
        ratio = (cb1 / table.loc[other, era_cols]).astype(float)
        print(f"  CB-1 / {other:<26}" + "  ".join(f"{v:6.2f}x" for v in ratio))

    print("\n=== Tag volume so với độ phổ biến ===")
    for k, v in tag_volume_vs_popularity().items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="full", choices=["dev", "full"])
    main(parser.parse_args().protocol)
