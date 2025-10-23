# EVユーザー行動パターン分析

このディレクトリには、EVユーザーの日次行動パターンを分析するためのコードが含まれています。

## 概要

EVユーザーの充電行動を最適化するために、日々の行動パターン（通勤・在宅・外出など）を自動的に分類・分析します。

### 主な特徴

- **06:00起点の日界処理**: 睡眠→通勤→帰宅→就寝の1サイクルを一日として扱う
- **時間×場所の滞在パターン**: 1時間ビンごとの場所クラスタ滞在比率をベクトル化
- **場所間遷移パターン**: 移動の流れ（自宅→職場→充電ステーション→自宅など）を特徴量化
- **柔軟なクラスタリング**: KMeansとHDBSCANの両方に対応
- **個別分析**: hashvinごとに行動パターンを抽出

## ファイル構成

```
eda_behavior_analysis/
├── behavior_analysis_pipeline.py  # メインモジュール
├── behavior_analysis.ipynb        # Jupyter Notebook（実行・可視化）
├── behavior_analysis要件.md        # 詳細要件定義
├── ev_sessions_1.csv              # サンプルデータ
└── README.md                      # 本ファイル
```

## 使い方

### 1. 基本的な使い方

```python
from behavior_analysis_pipeline import (
    BehaviorAnalysisConfig,
    DailyBehaviorVectorizer,
    BehaviorPatternClusterer,
    load_session_data
)

# 設定
config = BehaviorAnalysisConfig(
    hour_bin_size=1,                     # 1時間ビン
    topK_clusters=20,                    # 上位20クラスタ
    clustering_method='kmeans',          # 'kmeans' or 'hdbscan'
    include_transition_features=True,    # 遷移特徴量を含める
    topK_transitions=15                  # 上位15遷移パターン
)

# データ読み込み
df_sessions = load_session_data("ev_sessions_1.csv", min_duration_minutes=20)

# Step1: 日次行動ベクトル生成
vectorizer = DailyBehaviorVectorizer(config)
daily_vectors = vectorizer.fit_transform(df_sessions)

# Step2: 行動パターンクラスタリング
clusterer = BehaviorPatternClusterer(config)
for hashvin in daily_vectors.index.get_level_values('hashvin').unique():
    result = clusterer.fit_transform(daily_vectors, hashvin)
```

### 2. クラスタリング手法の選択

#### KMeans（デフォルト）
- 安定したクラスタ数を自動選択
- Silhouette/Calinski-Harabasz/Davies-Bouldinの3指標で最適k値を決定

```python
config = BehaviorAnalysisConfig(
    clustering_method='kmeans',
    k_range=(3, 10)  # 探索範囲
)
```

#### HDBSCAN（階層的密度ベース）
- クラスタ数を自動決定
- ノイズ点（異常な日）を自動検出

```python
config = BehaviorAnalysisConfig(
    clustering_method='hdbscan',
    hdbscan_min_cluster_size=5,  # 最小クラスタサイズ
    hdbscan_min_samples=3         # 最小サンプル数
)
```

**注意**: HDBSCANを使用する場合は事前にインストールが必要です：
```bash
pip install hdbscan
```

### 3. 特徴量の選択

#### 滞在特徴量のみ
時間帯×場所の滞在比率のみを使用：

```python
config = BehaviorAnalysisConfig(
    include_transition_features=False
)
```

#### 滞在＋遷移特徴量（推奨）
場所間の移動パターンも考慮：

```python
config = BehaviorAnalysisConfig(
    include_transition_features=True,
    topK_transitions=15  # 上位15遷移パターン
)
```

## 特徴量の詳細

### 滞在特徴量（Static Features）
- `ratio__{hour_bin}__{cluster}`: 時間帯×場所クラスタの滞在比率
  - 例: `ratio__08-09__I_101` = 08:00-09:00にI_101クラスタに滞在していた比率
- `ratio_OTHER__{hour_bin}`: 上位K以外のクラスタの合計滞在比率

### 遷移特徴量（Transition Features）
- `trans__{from_cluster}--{to_cluster}`: 特定の遷移パターンの発生回数
  - 例: `trans__I_101--I_202` = I_101からI_202への遷移回数
- `trans_count_morning/afternoon/evening/night`: 時間帯別の遷移数
- `trans_count_total`: 総遷移数
- `unique_clusters_visited`: 訪問した場所クラスタの種類数

## 出力データ

### daily_behavior_vectors.csv
- インデックス: `(hashvin, date06)`
- 滞在特徴量: 約500次元（24時間×20クラスタ）
- 遷移特徴量: 約20次元（15遷移パターン + 統計量）
- メタ情報: `total_minutes`, `weekday`, `is_empty_day`

### daily_vectors_with_clusters.csv
- 上記に加えて `day_pattern_cluster` カラムが追加
- クラスタID（0, 1, 2, ...）または -1（ノイズ/空日）

## 分析結果の活用

### 1. 行動パターンの解釈
- **通勤パターン**: 朝に自宅→職場、夕方に職場→自宅の遷移が多い
- **在宅パターン**: 一日中同じクラスタ（自宅様）に滞在
- **外出パターン**: 複数の異なるクラスタを訪問

### 2. 特徴量設計への還元
クラスタリング結果から「直前N時間」の特徴量を設計：

- `home_ratio_last6h`: 直前6時間で自宅様クラスタに滞在していた比率
- `work_ratio_last6h`: 直前6時間で職場様クラスタに滞在していた比率
- `unique_clusters_last3h`: 直前3時間で訪問したクラスタ数

### 3. 放置場所予測への活用
行動パターンと次回放置場所の関連を分析し、予測モデルの特徴量として活用。

## パラメータ調整ガイド

| パラメータ | 推奨値 | 調整の目安 |
|-----------|--------|----------|
| `hour_bin_size` | 1 | データが少ない場合は2-3時間に |
| `topK_clusters` | 20 | クラスタ数が多い場合は増やす |
| `topK_transitions` | 15 | 遷移パターンが多い場合は増やす |
| `k_range` (KMeans) | (3, 10) | データ量に応じて調整 |
| `hdbscan_min_cluster_size` | 5 | 小さいと細かいクラスタが増える |

## トラブルシューティング

### 警告: 行合計の最大偏差が許容範囲を超えています
- **原因**: セッションの重複や日跨ぎ処理の問題
- **対処**: 許容範囲（1e-3）内なら問題なし。大きい場合はデータを確認

### HDBSCANでクラスタが検出されない
- **原因**: データが均質すぎる、または`min_cluster_size`が大きすぎる
- **対処**: `min_cluster_size`を小さくする（3-5程度）

### メモリ不足エラー
- **原因**: 特徴量次元数が多い（大量のクラスタ×遷移パターン）
- **対処**: `topK_clusters`や`topK_transitions`を減らす

## 参考資料

- [behavior_analysis要件.md](behavior_analysis要件.md): 詳細な要件定義
- Jupyter Notebook: [behavior_analysis.ipynb](behavior_analysis.ipynb)

## ライセンス

このコードは機械学習研究用途で作成されています。


