"""Ánh xạ tên mô hình trong file config -> lớp cài đặt.

Mỗi phase tiếp theo chỉ cần thêm một dòng vào REGISTRY, không phải sửa runner.
"""

from src.models.base_recommender import BaseRecommender
from src.models.popularity_recommender import MostPopRecommender, RandomRecommender
from src.models.tfidf_content_model import TfidfContentModel
from src.models.genome_content_model import GenomeContentModel
from src.models.genome_ridge_user_model import GenomeRidgeUserModel
from src.models.itemknn_cf_model import ItemKnnCfModel
from src.models.als_cf_model import AlsCfModel
from src.models.hybrid_content_model import HybridLateFusionModel
from src.models.bm25_content_model import Bm25ContentModel
from src.models.rs_itemcf_model import RsItemCfModel
from src.models.rs_bpr_model import RsBprModel

REGISTRY: dict[str, type[BaseRecommender]] = {
    "random": RandomRecommender,
    "mostpop": MostPopRecommender,
    "cb1-tfidf": TfidfContentModel,
    "cb3-genome": GenomeContentModel,
    "cb3-ridge": GenomeRidgeUserModel,
    "cf-itemknn": ItemKnnCfModel,
    "cf-als": AlsCfModel,
    "cb4-hybrid": HybridLateFusionModel,
    "cb2-bm25": Bm25ContentModel,
    "rs-itemcf": RsItemCfModel,
    "rs-bpr": RsBprModel,
}


def build_model(name: str, params: dict | None = None) -> BaseRecommender:
    """Khởi tạo mô hình theo tên và tham số lấy từ config."""
    if name not in REGISTRY:
        raise KeyError(f"Mô hình '{name}' chưa đăng ký. Hiện có: {sorted(REGISTRY)}")
    return REGISTRY[name](**(params or {}))
