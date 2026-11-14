"""Fixture và helper dùng chung cho test Phase 2."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src import paths
from src.data.build_interaction_matrices import TrainData
from src.models.vector_normalization import safe_l2_normalize


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Cache sim/SVD phải ghi vào tmp, không làm bẩn data/processed thật."""
    monkeypatch.setattr(paths, "PROCESSED_DIR", tmp_path)


@pytest.fixture
def mock_train():
    return _mock_train


@pytest.fixture
def unit_rows():
    return _unit_rows


def _mock_train(n_users=10, n_items=20, n_per_user=5) -> TrainData:
    rng = np.random.default_rng(42)
    rows, cols, data = [], [], []
    for u in range(n_users):
        for i in rng.choice(n_items, size=n_per_user, replace=False):
            rows.append(u)
            cols.append(int(i))
            data.append(float(rng.choice([1.0, 2.0, 3.5, 4.0, 5.0])))
    csr = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype="float32")
    csr.sort_indices()
    return TrainData(train_csr=csr, train_df=pd.DataFrame({"u": rows, "i": cols, "r": data}),
                     n_users=n_users, n_items=n_items)


def _unit_rows(n, d, seed=0) -> np.ndarray:
    return safe_l2_normalize(np.random.default_rng(seed).standard_normal((n, d)).astype("float32"))
