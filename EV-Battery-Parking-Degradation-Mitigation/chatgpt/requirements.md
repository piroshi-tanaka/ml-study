# EV電池放置劣化軽減 - 5セッション判定

## 概要
時系列データに5つのセッションタイプを判定・付与し、EV電池の放置劣化分析を実施。

## 5つのセッションタイプ
1. **充電セッション** - `is_charging_stitched = True`
2. **移動中セッション** - `is_moving = True` (IG-ONかつ移動)
3. **アイドリングセッション** - `is_idling = True` (IG-ONかつ非移動)
4. **パーキングセッション** - `is_parking = True` (IG-OFF)
5. **充電後移動なし放置** - `is_no_move_after_charge = True` (分析対象外)

## 基本ルール
- **排他的分類**: 各時点で1つのセッションのみがTrue
- **充電優先**: 充電中は他のセッションより優先
- **移動判定**: 距離閾値（150m）で移動/非移動を判定

## 使用方法

### 基本的な使用例
```python
from session_detector import detect_sessions, SessionParams, validate_sessions

# パラメータ設定
params = SessionParams(
    DIST_TH_m=150.0,      # 移動判定距離閾値（m）
    PARK_TH_min=360.0,    # 長時間放置閾値（分）= 6時間
    GAP_MAX_min=15.0,     # 充電ギャップ許容時間（分）
    WINDOW_min=60.0       # 充電後移動チェック時間窓（分）
)

# セッション判定実行
result = detect_sessions(df, params)

# 品質チェック
quality = validate_sessions(result)
```

### 必要な入力データ
```
必須カラム:
- hashvin: 車両ID
- tsu_current_time: 観測時刻
- tsu_igon_time: IG-ON時刻（IG-OFF時はNaN）
- tsu_latitude, tsu_longitude: 位置情報
- soc: バッテリー残量（0-100%）
- charge_mode: 充電モード
```

### 出力カラム
```
セッションフラグ:
- is_charging_stitched: 充電セッション
- is_moving: 移動中セッション  
- is_idling: アイドリングセッション
- is_parking: パーキングセッション
- is_no_move_after_charge: 充電後移動なし（除外対象）

セッションタイプ（改善版）:
- session_type: 主要セッションタイプ（境界があっても主要セッションを優先）
  - "charging": 充電中
  - "moving": 移動中
  - "idling": アイドリング（放置の一種）
  - "parking": パーキング（放置の一種）
  - "inactive": 非活動状態（idle + parking の総称）

放置状態統合:
- is_inactive: アイドリング or パーキング（放置状態の統合フラグ）
- inactive_start_flag, inactive_end_flag: 放置開始・終了フラグ
- inactive_session_id: 放置セッションID

境界情報（新機能）:
- boundary_events: セッション境界イベント（パイプ区切り）
  - 例: "charge_end|movement_start", "parking_end"
- has_boundary: 境界イベントがあるかのboolean

開始・終了フラグ:
- charge_start_flag, charge_end_flag
- movement_start_flag, movement_end_flag
- idling_start_flag, idling_end_flag  
- parking_start_flag, parking_end_flag

セッションID:
- charge_session_id, movement_session_id
- idling_session_id, parking_session_id

補助情報:
- distance_prev_m: 前観測点からの距離（sklearnのhaversine使用）
- parking_category: 放置時間分類
- parking_duration_min: 放置時間（分）

統合セッション抽出（新機能）:
- extract_charge_to_inactive_sessions(): 充電→移動→放置（inactive）の統合セッション
- get_session_data_by_boundary(): 特定境界イベントのデータ抽出
- is_excluded_no_movement: 充電後移動なしで除外対象かの判定

セッション詳細データ抽出（最新機能）:
- extract_session_details(): 各セッションの開始・終了時点での詳細データを抽出
  - session_type: セッションタイプ（charging, moving, idling, parking, inactive）
  - event_type: 'start' または 'end'
  - timestamp: 開始・終了時刻
  - soc: SOC値
  - distance_from_prev: 前の時点からの移動距離
  - duration_min: セッション時間（終了時点のみ）
  - soc_diff: SOC差分（終了時点のみ）
  - total_distance: セッション中の総移動距離（終了時点のみ）
  - inactive_breakdown: 放置の内訳（idle|parking）
- get_session_summary_by_type(): セッションタイプ別サマリー取得
```

## パラメータ

```python
@dataclass
class SessionParams:
    DIST_TH_m: float = 150.0        # 移動判定距離閾値（m）
    SOC_TH_pct: float = 5.0         # SOC変化閾値（%）
    PARK_TH_min: float = 360.0      # 長時間放置閾値（分）= 6時間
    GAP_MAX_min: float = 15.0       # 充電ギャップ許容時間（分）
    WINDOW_min: float = 60.0        # 充電後移動チェック時間窓（分）
    SHORT_PARK_min: float = 180.0   # 短時間放置上限（分）= 3時間
    LONG_PARK_min: float = 360.0    # 長時間放置下限（分）= 6時間
```

## 実装ファイル

- **session_detector.py**: メインモジュール（5セッション判定）
- **usage_example.py**: 使用例
- **requirements.md**: 本ファイル（仕様書）