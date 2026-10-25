# 协同过滤实现指南

## 概述

本项目实现了两种协同过滤推荐算法:
1. **Item-based CF** - 基于物品相似度的协同过滤
2. **ALS (Alternating Least Squares)** - 交替最小二乘矩阵分解

## 项目结构

```
RecommendationSystem/
├── src/
│   ├── recommenders/
│   │   ├── __init__.py
│   │   ├── item_cf.py          # Item-based CF实现
│   │   └── als_recommender.py  # ALS实现
│   ├── evaluation/              # 评估pipeline
│   └── data_loader.py           # 数据加载
├── notebooks/
│   ├── 01-popularity-based.ipynb
│   └── 02-cf-experiments.ipynb  # CF实验notebook
├── scripts/
│   └── export_models.py         # 模型导出脚本
├── models/                      # 导出的模型文件
└── data/
    ├── raw/ml-25m/             # MovieLens 25M原始数据
    └── processed/               # 预处理后的数据
```

## 数据准备

### 训练数据格式
- **显式反馈**: `train_ml25m.csv` (userId, movieId, rating, timestamp)
- **隐式反馈**: 由 `DataPreprocessor.to_implicit()` 转换,阈值=4.0

### 数据加载示例

```python
from src.data_loader import ML25MLoader, DataPreprocessor

# 加载预处理数据
train_df = pd.read_csv('data/processed/train_ml25m.csv')
test_df = pd.read_csv('data/processed/test_ml25m.csv')

# 构建user-item稀疏矩阵
preprocessor = DataPreprocessor()
user_item_csr, user_map, movie_map = preprocessor.build_user_item_matrix(
    train_df, 
    value_column='rating'
)

# 转换为隐式反馈
train_implicit = preprocessor.to_implicit(train_df, threshold=4.0)
```

## Item-based CF

### 原理
- 计算物品间余弦相似度矩阵
- 对用户历史交互物品,推荐最相似的物品

### 使用方法

```python
from src.recommenders import ItemCFRecommender

# 训练
item_cf = ItemCFRecommender()
item_cf.fit(user_item_csr, user_map, movie_map)

# 推荐
user_id = 123
seen_items = set(train_df[train_df['userId'] == user_id]['movieId'])
recommendations = item_cf.recommend(user_id, n=10, seen=seen_items)

# 保存/加载
item_cf.save('models/item_cf.pkl')
loaded_model = ItemCFRecommender.load('models/item_cf.pkl')
```

### 关键参数
- 无显式超参数(基于固定的余弦相似度)
- 推荐时自动过滤用户已交互物品

## ALS推荐器

### 原理
- 矩阵分解: `R ≈ X·Y^T` (用户因子 × 物品因子)
- 交替优化,适合隐式反馈(confidence = 1 + alpha * rating)

### 使用方法

```python
from src.recommenders import ALSRecommender

# 训练 (隐式反馈)
als = ALSRecommender(
    factors=64,           # 潜在因子数
    regularization=0.01,  # L2正则化
    iterations=15,        # 迭代次数
    alpha=1.0,            # 置信度权重
    random_state=42
)
als.fit(train_implicit)

# 推荐
recommendations = als.recommend(user_id, n=10, seen=seen_items)

# 保存/加载
als.save('models/als_best.pkl')
loaded_als = ALSRecommender.load('models/als_best.pkl')
```

### 超参数调优

在 `notebooks/02-cf-experiments.ipynb` 中实现了网格搜索:

```python
param_grid = {
    'factors': [32, 64, 128],
    'regularization': [0.01, 0.05, 0.1],
    'iterations': [15, 25]
}

# 在验证集上搜索最优参数组合
# 选择NDCG@10最高的配置
```

**推荐起点**:
- `factors=64, regularization=0.01, iterations=15, alpha=1.0`

## 评估与比较

### 评估指标
- **Precision@K**: Top-K推荐的准确率
- **Recall@K**: Top-K推荐覆盖的相关物品比例
- **NDCG@K**: 归一化折扣累积增益(考虑排序)
- **Hit Rate@K**: 至少命中1个相关物品的用户比例

### Baseline对比

```python
from src.evaluation import EvaluationPipeline
from src.popularity_baselines import PopularityRecommender, BayesianPopularityRecommender

# 评估CF模型
evaluator = EvaluationPipeline(train_df, test_df, positive_threshold=4.0)
item_cf_results = evaluator.evaluate_model(item_cf, model_name="ItemCF")
als_results = evaluator.evaluate_model(als, model_name="ALS-best")

# 对比Popularity baseline
pop_model = BayesianPopularityRecommender()
pop_model.fit(train_df)
pop_results = evaluator.evaluate_model(pop_model, model_name="Popularity-Weighted")

# 合并结果
all_results = pd.DataFrame([pop_results, item_cf_results, als_results])
```

### Ground Truth说明
- **来源**: `test_df` 中 `rating >= 4.0` 的物品
- **过滤**: 跳过训练集中未出现的用户(cold-start)
- **目的**: 模拟真实场景下用户会喜欢的物品

## 模型导出

### 方式1: Notebook导出

```python
# 在 02-cf-experiments.ipynb 最后一节
models_dir = PROJECT_ROOT / 'models'
models_dir.mkdir(exist_ok=True)

item_cf.save(str(models_dir / 'item_cf_best.pkl'))
als_best.save(str(models_dir / 'als_best.pkl'))
pop_weighted.save(str(models_dir / 'popularity_weighted.pkl'))
```

### 方式2: 命令行脚本

```bash
python scripts/export_models.py
```

脚本会:
1. 加载 `data/processed/` 的训练数据
2. 训练 ItemCF、ALS、Bayesian Popularity 模型
3. 导出到 `models/` 目录

**输出文件**:
- `models/item_cf_best.pkl`
- `models/als_best.pkl`
- `models/popularity_weighted.pkl`

### 生产环境加载

```python
from src.recommenders import ItemCFRecommender, ALSRecommender

# 加载已训练模型
model = ALSRecommender.load('models/als_best.pkl')

# 直接推荐 (需要传入seen items避免重复推荐)
user_id = 456
seen = get_user_history(user_id)  # 从数据库获取
recs = model.recommend(user_id, n=10, seen=seen)
```

## Case Study示例

在notebook的Section 7实现了推荐案例分析:

```python
def show_recommendations(user_id, models_dict, n=10):
    """展示不同模型的推荐结果"""
    # 获取用户历史
    seen = set(train_df[train_df['userId'] == user_id]['movieId'].values)
    
    # 对比各模型推荐
    for model_name, model in models_dict.items():
        recs = model.recommend(user_id, n=n, seen=seen)
        # 显示推荐物品标题和类型

models_dict = {
    'Popularity': pop_weighted,
    'ItemCF': item_cf,
    'ALS-best': als_best
}

# 分析3个样本用户
for user_id in [10, 50, 100]:
    show_recommendations(user_id, models_dict)
```

## 性能对比结果

根据 `02-cf-experiments.ipynb` 的实验:

| 模型 | NDCG@10 | Recall@10 | Precision@10 |
|------|---------|-----------|--------------|
| Popularity-Weighted | ~0.20 | ~0.08 | ~0.06 |
| ItemCF | **~0.28** | **~0.12** | **~0.09** |
| ALS-best | **~0.30** | **~0.13** | **~0.10** |

**结论**:
- CF方法显著优于popularity baseline (~50%提升)
- ALS略优于ItemCF (特别在NDCG和Recall上)
- 两者都能提供个性化推荐

## 常见问题

### Q1: ALS为什么需要隐式反馈?
**A**: ALS优化的是加权误差,`confidence = 1 + alpha * rating`。对于评分数据,高分=高置信度正样本,低分/无交互=低置信度负样本。

### Q2: ItemCF和ALS如何选择?
**A**: 
- **ItemCF**: 可解释性强(基于相似物品),适合冷启动物品
- **ALS**: 性能更好,适合大规模数据,需要GPU加速

### Q3: 如何处理新用户/新物品?
**A**: 
- 新用户: 降级到popularity推荐
- 新物品: ItemCF无法推荐,ALS可通过物品特征扩展

### Q4: 模型多久需要重新训练?
**A**: 
- 每日/每周增量更新(追加新交互)
- 定期全量重训练(1-2周)

## 依赖库

```
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
implicit>=0.7.0  # ALS实现
matplotlib>=3.7.0
seaborn>=0.12.0
```

安装: `pip install -r requirements.txt`

## 参考资料

- [Item-based CF论文](https://dl.acm.org/doi/10.1145/371920.372071)
- [Implicit库文档](https://implicit.readthedocs.io/)
- [MovieLens数据集](https://grouplens.org/datasets/movielens/)
