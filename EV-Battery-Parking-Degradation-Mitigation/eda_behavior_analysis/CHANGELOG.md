# 変更履歴

## 2025-10-23 - メジャーアップデート

### 🎯 主な改善

#### 1. ファイル構成の最適化
**変更前:** 単一ファイル `behavior_analysis_pipeline.py` (864行)

**変更後:** モジュール化された構成
```
config.py              (52行) - 設定パラメータ
time_processor.py     (119行) - 時間範囲処理
vectorizer.py         (358行) - 日次行動ベクトル生成
clusterer.py          (272行) - クラスタリング
map_visualizer.py     (526行) - 地図可視化
utils.py               (24行) - ユーティリティ
behavior_analysis_pipeline.py (22行) - 統合インターフェース
```

**メリット:**
- ✅ 可読性・保守性の向上
- ✅ 再利用性の改善
- ✅ テストの容易化
- ✅ 責任の明確化（単一責任原則）

#### 2. 地図可視化の改善

##### 変更1: 一日ごとの表示（前日データを自動削除）
**変更前:** タイムスライダーで累積表示（過去のデータも残る）

**変更後:** スライダーで切り替えた日のみ表示
```python
# 前日/次日ボタン + スライダーで日付を切り替え
# 表示される日のデータのみマップ上に表示
# 前日のマーカーは自動的に消える
```

**実装方法:**
- 日付ごとにFeatureGroupを作成
- JavaScriptで日付切り替え時に古いレイヤーを削除
- カスタムスライダーコントロール追加

##### 変更2: クラスタIDを常時表示
**変更前:** ポップアップを開かないとクラスタIDが見えない

**変更後:** マーカー下に黄色ラベルでクラスタIDを表示
```
  ⚡ or 🅿️  ← セッションタイプアイコン
  ┌──────┐
  │ I_0  │  ← クラスタID（常時表示）
  └──────┘
```

**実装方法:**
- `folium.DivIcon` を使用してカスタムマーカー作成
- HTMLとCSSで2段階表示（アイコン + ラベル）
- 黄色背景で視認性を向上

### 🔧 技術的な変更

#### インポート構造
```python
# 変更前
from behavior_analysis_pipeline import (
    BehaviorAnalysisConfig,
    TimeRangeProcessor,
    DailyBehaviorVectorizer,
    BehaviorPatternClusterer,
    load_session_data
)

# 変更後（同じインターフェース維持）
from behavior_analysis_pipeline import (
    BehaviorAnalysisConfig,
    TimeRangeProcessor,
    DailyBehaviorVectorizer,
    BehaviorPatternClusterer,
    load_session_data
)
# 内部的にはモジュールから再エクスポート
```

**後方互換性:** ✅ 既存のコードはそのまま動作

#### 型アノテーションの追加
```python
# 主要な型アノテーションを追加
self.top_clusters_: List[str] = []
self.top_transitions_: List[Tuple[str, str]] = []
bin_cluster_minutes: Dict[Tuple[str, str], float] = {}
```

### 📊 地図の新機能

#### 日付切り替えUI
```
┌──────────────────────────────────────┐
│        2025-10-01                    │  ← 現在表示中の日付
├──────────────────────────────────────┤
│ ◀前日 ━━━●━━━━━━━━━━━ 次日▶ │  ← スライダー
│           1 / 30                     │  ← 進捗表示
└──────────────────────────────────────┘
```

#### マーカー表示
- **青色マーカー (🅿️)**: 放置セッション
- **赤色マーカー (⚡)**: 充電セッション
- **緑色ポリライン**: 移動セッション
- **黄色ラベル**: クラスタID（常時表示）

#### ポップアップ詳細
```
┌─────────────────────────────────┐
│  ⚡ 充電セッション #42          │
├─────────────────────────────────┤
│  クラスタID: C_1                │  ← 黄色ハイライト
├─────────────────────────────────┤
│  開始時刻: 2025-10-01 18:30    │
│  終了時刻: 2025-10-01 20:15    │
│  滞在時間: 1時間45分            │
│  開始SOC: 35.5%                 │
│  終了SOC: 78.2%                 │
│  SOC変化: 📈 +42.7%             │
│  位置: 35.681236, 139.767125    │
├─────────────────────────────────┤
│  📍 Google Mapsで開く           │  ← クリック可能
└─────────────────────────────────┘
```

### 📝 ドキュメント

新規追加:
- `README.md` - 完全な使用ガイド
- `CHANGELOG.md` - この変更履歴

更新:
- `behavior_analysis要件.md` - Step3の詳細追加

### 🐛 バグ修正

- タイムゾーン処理の改善（混在形式対応）
- 浮動小数点精度チェックの緩和
- Unicode文字エンコーディングエラーの修正
- 型アノテーションの追加

### 🚀 パフォーマンス

- ファイル分割により、必要なモジュールのみインポート可能
- 地図のレイヤー管理最適化（日ごとに分割）

### 📦 互換性

- **Python**: 3.8+
- **依存パッケージ**: 変更なし
  - pandas, numpy, scikit-learn
  - matplotlib, seaborn
  - folium
  - hdbscan (オプション)

### 🔄 マイグレーションガイド

既存のコードは**変更不要**です。以下はそのまま動作します：

```python
# 既存のコード（変更不要）
from behavior_analysis_pipeline import (
    BehaviorAnalysisConfig,
    DailyBehaviorVectorizer,
    BehaviorPatternClusterer
)

config = BehaviorAnalysisConfig()
vectorizer = DailyBehaviorVectorizer(config)
# ... 以下同様
```

新しい地図機能を使用する場合：

```python
# 新規追加
from map_visualizer import BehaviorMapVisualizer

visualizer = BehaviorMapVisualizer(
    lat_col='start_lat',
    lon_col='start_lon'
)

behavior_map = visualizer.create_behavior_map(
    sessions_df=df_sessions,
    hashvin='hv_0001',
    daily_vectors_with_clusters=df_clustered,
    output_path='outputs/map.html'
)
```

### ✨ 次のステップ

今後の改善案：
- [ ] PCA次元削減の可視化
- [ ] クラスタ間の遷移確率マトリックス
- [ ] 予測モデルの統合（次回放置場所予測）
- [ ] ダッシュボードUI（Streamlit/Dash）
- [ ] リアルタイム分析対応

---

**Author**: ML Engineer  
**Date**: 2025-10-23  
**Version**: 2.0.0



