# 协同过滤快速启动指南

## ✅ 实现完成

已完成 **Item-based CF** 和 **ALS (Alternating Least Squares)** 推荐算法的实现、评估和导出。

## 📂 新增文件

```
src/recommenders/
├── __init__.py
├── item_cf.py          # Item-based CF实现
└── als_recommender.py  # ALS实现

notebooks/
└── 02-cf-experiments.ipynb  # 完整实验流程

scripts/
└── export_models.py    # 命令行模型导出

docs/
├── cf-implementation-guide.md  # 详细实现文档
└── cf-quickstart.md           # 本文档

tests/
└── test_recommenders.py  # 基础测试
```

## 🚀 快速使用

### 1. 运行实验 Notebook

```bash
jupyter notebook notebooks/02-cf-experiments.ipynb
```

**包含内容**:
- Item-based CF训练与评估
- ALS基线模型
- 网格搜索超参数调优
- 与Popularity Baseline对比
- 可视化分析
- 案例推荐展示
- 模型导出

### 2. 命令行导出模型

```bash
python scripts/export_models.py
```

**生成文件**:
- `models/item_cf_best.pkl`
- `models/als_best.pkl`
- `models/popularity_weighted.pkl`

### 3. 生产环境加载模型

```python
from src.recommenders import ItemCFRecommender, ALSRecommender

# 加载训练好的模型
als = ALSRecommender.load('models/als_best.pkl')

# 推荐 (需要过滤用户已交互物品)
user_id = 123
seen_items = get_user_history(user_id)  # 从数据库获取
recommendations = als.recommend(user_id, n=10, seen=seen_items)
```

## 📊 Notebook结构

`notebooks/02-cf-experiments.ipynb` 包含完整工作流:

1. **环境设置**: 导入库和定义路径
2. **加载数据**: 读取预处理的训练/测试集
3. **Item-based CF**: 训练并评估
4. **ALS基线**: 使用默认参数 (factors=64, reg=0.01, iter=15)
5. **超参数调优**: 网格搜索最佳参数
6. **最佳ALS模型**: 用最优参数在全量训练集上重新训练
7. **Baseline对比**: 与Popularity方法比较
8. **可视化**: 柱状图和雷达图展示指标
9. **案例分析**: 展示3个用户的推荐结果
10. **模型导出**: 保存最佳模型到 `models/` 目录

## 💾 模型导出

### 方法1: Notebook内导出

在 `notebooks/02-cf-experiments.ipynb` 最后一节运行:

```python
item_cf.save('models/item_cf_best.pkl')
als_best.save('models/als_best.pkl')
pop_weighted.save('models/popularity_weighted.pkl')
```

### 方法2: 命令行脚本

```bash
python scripts/export_models.py
```

自动执行:
- 加载 `data/processed/` 数据
- 训练 Item-CF、ALS、Bayesian Popularity
- 保存到 `models/` 目录
- 打印保存路径

## 🔄 生产环境使用

```python
from src.recommenders import ALSRecommender

# 一次性加载
model = ALSRecommender.load('models/als_best.pkl')

# 为多个用户推荐
def recommend_for_user(user_id):
    seen = get_user_interactions(user_id)  # 从数据库/缓存获取
    return model.recommend(user_id, n=10, seen=seen)
```

**注意**:
- `seen` 参数必须是 `Set[int]`,包含用户已交互的物品ID
- 推荐结果是 `List[int]`,按分数降序排列
- 如果用户在训练集中不存在,ALS会返回全局流行物品

## 📈 预期性能 (MovieLens-25M test集)

| 模型 | Precision@10 | Recall@10 | NDCG@10 | Hit Rate@10 |
|------|--------------|-----------|---------|-------------|
| **ALS (tuned)** | ~0.08 | ~0.12 | ~0.15 | ~0.45 |
| **Item-CF** | ~0.06 | ~0.10 | ~0.12 | ~0.40 |
| Popularity (Bayesian) | ~0.04 | ~0.06 | ~0.08 | ~0.30 |

*实际数值取决于数据预处理方式和train/test划分*

## ⚙️ 推荐超参数

### Item-CF
- **K (相似物品数)**: 20-50 (代码中默认用所有相似物品)
- **相似度指标**: Cosine similarity
- **归一化**: L2正则化物品向量

### ALS
**基线参数** (快速测试):
- `factors=64`, `regularization=0.01`, `iterations=15`, `alpha=1.0`

**调优后参数** (网格搜索结果):
- `factors`: 在 32/64/128 中选择
- `regularization`: 在 0.01/0.05/0.1 中选择
- `iterations`: 在 15/25 中选择

**调优方法**:
1. 在验证集 (train的10%子集) 上网格搜索
2. 选择NDCG@10最高的参数
3. 用最优参数在全量训练集上重新训练

## 📖 详细文档

完整实现细节、API文档、设计思路请参考:
- **实现指南**: `docs/cf-implementation-guide.md`

## ✅ 已实现功能

- [x] Item-based CF (cosine similarity)
- [x] ALS (implicit feedback)
- [x] 模型持久化 (save/load)
- [x] 与Popularity baseline对比
- [x] 超参数网格搜索
- [x] 完整评估 (Precision/Recall/NDCG/Hit Rate)
- [x] 可视化分析
- [x] 案例推荐展示

## 🔜 后续工作

可能的扩展方向:
1. **Deep Learning方法**: Neural CF, Wide & Deep
2. **内容特征融合**: 利用电影类型、标签
3. **时间动态**: 时序推荐模型
4. **冷启动处理**: 针对新用户/物品的策略
5. **在线评估**: A/B测试框架

## 🐛 常见问题

### Q: 运行notebook时报 `ModuleNotFoundError`
**A**: 确保安装所有依赖:
```bash
pip install -r requirements.txt
```

### Q: `export_models.py` 运行很慢
**A**: ALS训练在全量数据上需要5-10分钟。可以先在notebook中验证逻辑。

### Q: 推荐结果全是热门物品
**A**: 检查 `seen` 参数是否正确传递,并确保用户在训练集中有足够的交互记录。

### Q: 如何更新模型?
**A**: 重新运行训练脚本并覆盖旧模型文件。建议保留版本号 (如 `als_v2.pkl`)。

---

**实现完成时间**: 2026-09-10  
**文档维护**: 协同过滤实现团队
