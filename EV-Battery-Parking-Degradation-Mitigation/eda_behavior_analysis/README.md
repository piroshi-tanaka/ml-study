# EV行動パターン分析パイプライン

EVユーザーの行動パターンを分析し、次回の長期放置場所とΔSOCを予測するためのデータ分析パイプラインです。

## 📁 ファイル構成

```
eda_behavior_analysis/
├── README.md                        # このファイル
├── behavior_analysis要件.md         # 詳細要件定義
│
├── config.py                        # 設定パラメータ（BehaviorAnalysisConfig）
├── time_processor.py                # 時間範囲処理（06:00起点の日界処理）
├── vectorizer.py                    # Step1: 日次行動ベクトル生成
├── clusterer.py                     # Step2: 行動パターンクラスタリング
├── map_visualizer.py                # Step3: 地図可視化（Folium）
├── utils.py                         # ユーティリティ関数
│
├── behavior_analysis_pipeline.py   # パイプライン統合モジュール（再エクスポート）
└── behavior_analysis.ipynb         # メイン実行ノートブック
```

## 🚀 使い方

### 1. 環境構築

```bash
pip install pandas numpy scikit-learn matplotlib seaborn folium hdbscan
```

### 2. データ準備

以下のCSVファイルを準備：
- `ev_sessions_1.csv`: クラスタリング済みセッションデータ
- `ev_sessions_0.csv`: 元データ（地図可視化用、movingセッション含む）

### 3. Jupyter Notebookで実行

`behavior_analysis.ipynb` を開いて実行してください。

#### Step 0: 設定

```python
from behavior_analysis_pipeline import (
    BehaviorAnalysisConfig,
    DailyBehaviorVectorizer,
    BehaviorPatternClusterer,
    load_session_data
)

# 設定
config = BehaviorAnalysisConfig(
    hour_bin_size=1,                     # 時間ビンサイズ（時間）
    topK_clusters=20,                    # 上位K個のクラスタを使用
    topK_transitions=15,                 # 上位K個の遷移パターンを使用
    clustering_method='kmeans',          # 'kmeans' or 'hdbscan'
    include_transition_features=True,    # 遷移特徴量を含める
    k_range=(3, 10),                     # KMeansのk値探索範囲
    random_state=42
)
```

#### Step 1: 日次行動ベクトル生成

```python
# データ読み込み
df_sessions = load_session_data('ev_sessions_1.csv')

# 日次行動ベクトル生成
vectorizer = DailyBehaviorVectorizer(config)
daily_vectors = vectorizer.fit_transform(df_sessions)
```

**出力:**
- `daily_vectors`: hashvin × date06 の行動ベクトル
- 滞在特徴量: `ratio__{hour_bin}__{cluster}`
- 遷移特徴量: `trans__{from}--{to}`, `trans_count_{period}`

#### Step 2: 行動パターンクラスタリング

```python
# hashvinごとにクラスタリング
clusterer = BehaviorPatternClusterer(config)

results = {}
for hashvin in daily_vectors.index.get_level_values('hashvin').unique():
    df_clustered = clusterer.fit_transform(daily_vectors, hashvin)
    results[hashvin] = df_clustered
```

**出力:**
- 各日に行動パターンクラスタID（`day_pattern_cluster`）が付与される
- KMeansまたはHDBSCANで自動クラスタリング

#### Step 3: 地図可視化

```python
from map_visualizer import BehaviorMapVisualizer

# 元データ読み込み（movingセッション含む）
df_sessions_0 = pd.read_csv('ev_sessions_0.csv')

# 可視化
visualizer = BehaviorMapVisualizer(
    lat_col='start_lat',
    lon_col='start_lon',
    end_lat_col='end_lat',  # moving用
    end_lon_col='end_lon'   # moving用
)

# 地図作成
test_hashvin = 'hv_0001_demo'
behavior_map = visualizer.create_behavior_map(
    sessions_df=df_sessions_0,
    hashvin=test_hashvin,
    daily_vectors_with_clusters=results[test_hashvin],
    output_path=f'outputs/behavior_map_{test_hashvin}.html'
)
```

**地図の機能:**
- ⏱️ **タイムスライダー**: 日付を前後に切り替え（前日データは消える）
- 🏷️ **クラスタIDラベル**: マーカーに常時表示
- 🅿️ **青色マーカー**: 放置セッション
- ⚡ **赤色マーカー**: 充電セッション
- 🚗 **緑色ポリライン**: 移動セッション
- 📍 **GoogleMapsリンク**: ポップアップから直接ジャンプ

## 📊 出力データ

### 日次行動ベクトル (`daily_vectors`)

| カラム | 説明 |
|--------|------|
| `hashvin` | 車両ID（インデックス） |
| `date06` | 06:00起点の日付（インデックス） |
| `ratio__{hour_bin}__{cluster}` | 時間帯×クラスタの滞在比率 |
| `ratio_OTHER__{hour_bin}` | その他クラスタの滞在比率 |
| `trans__{from}--{to}` | 遷移パターン出現回数 |
| `trans_count_{period}` | 時間帯別遷移数 |
| `trans_count_total` | 総遷移数 |
| `unique_clusters_visited` | ユニーククラスタ訪問数 |
| `total_minutes` | 総滞在時間 |
| `weekday` | 曜日（0=月曜） |
| `is_empty_day` | 空日フラグ |

### クラスタリング結果 (`results[hashvin]`)

日次行動ベクトルに以下が追加：
- `day_pattern_cluster`: 行動パターンクラスタID

## 🎯 主要機能

### 1. 06:00起点の日界処理

睡眠周期を考慮し、06:00を日の境界として定義。
- 2025-10-23 20:00 → 2025-10-23 06:00の日
- 2025-10-23 03:00 → 2025-10-22 06:00の日

### 2. セッション分割

日をまたぐセッションを自動分割し、各日に按分。

### 3. 時間ビン按分

1時間（設定可能）ごとに滞在時間を按分。

### 4. 上位クラスタ選定

滞在時間の多い上位K個のクラスタに焦点を当て、残りは`OTHER`に集約。

### 5. 遷移特徴量

場所間の移動パターンを特徴量化：
- 上位K個の遷移パターンの出現回数
- 時間帯別の遷移数（朝/昼/夕/夜）
- 総遷移数、ユニーククラスタ訪問数

### 6. 自動クラスタリング

**KMeans:**
- Silhouette、Calinski-Harabasz、Davies-Bouldinスコアで最適k値を自動選択

**HDBSCAN:**
- クラスタ数を自動検出
- ノイズ点の識別

### 7. インタラクティブ地図

**日付切り替え:**
- 前日/次日ボタン
- スライダーで直接選択
- **前日データは自動的に消える**

**マーカー表示:**
- **クラスタIDが常時表示**（黄色ラベル）
- ポップアップで詳細情報
- GoogleMapsリンク

## ⚙️ 設定オプション

### BehaviorAnalysisConfig

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `hour_bin_size` | 1 | 時間ビンサイズ（時間） |
| **`use_topk_per_hour`** | **True** | **時間帯ごとTOP-K方式を使用（推奨）** |
| **`topk_per_hour`** | **3** | **各時間帯のTOP-Kクラスタ数** |
| `topK_clusters` | 20 | 上位クラスタ数（従来方式用） |
| `topK_transitions` | 15 | 上位遷移パターン数 |
| `clustering_method` | 'kmeans' | 'kmeans' or 'hdbscan' |
| `k_range` | (3, 10) | KMeansのk値探索範囲 |
| `hdbscan_min_cluster_size` | 5 | HDBSCANの最小クラスタサイズ |
| `hdbscan_min_samples` | 3 | HDBSCANの最小サンプル数 |
| `include_transition_features` | True | 遷移特徴量を含めるか |
| `day_start_hour` | 6 | 日の開始時刻 |
| `random_state` | 42 | 乱数シード |
| `timezone` | 'Asia/Tokyo' | タイムゾーン |

## 📈 可視化例

Jupyter Notebookでは以下を可視化：
- クラスタごとの平均行動ヒートマップ
- 曜日分布
- 代表日の詳細分析
- インタラクティブ地図（HTML出力）

## 🔍 トラブルシューティング

### ImportError: No module named 'hdbscan'

```bash
pip install hdbscan
```

### 地図が表示されない

- `outputs/behavior_map_{hashvin}.html` をブラウザで直接開いてください
- JavaScriptが有効になっているか確認してください

### クラスタIDが「N/A」と表示される

- `ev_sessions_1.csv` に `session_cluster` カラムがあるか確認
- または `daily_vectors_with_clusters` を正しく渡しているか確認

## 📝 更新履歴

- **2025-10-23**: 初版作成
  - Step1: 日次行動ベクトル生成
  - Step2: 行動パターンクラスタリング
  - Step3: 地図可視化（Folium）
  - 遷移特徴量の追加
  - HDBSCAN対応
  - ファイル分割によるコード整理
  - 地図のクラスタIDラベル表示
  - 一日ずつの地図表示（前日データ自動削除）

## 📧 お問い合わせ

質問や不明点があれば、開発者にお問い合わせください。
