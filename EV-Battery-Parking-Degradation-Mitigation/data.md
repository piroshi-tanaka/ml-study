# 放置抽出ハイブリッド案：最終整理（改訂版）

## 0. 前提
- 観測は **IG-ON時のみ**（IG-OFF中は欠測）
- 充電中は `is_charging=True`、充電停車は **放置対象外**
- しきい値（初期設定）
  - 距離 `DIST_TH = 150m`
  - SOC変化 `SOC_TH = 5%`
  - 放置時間 `PARK_TH = 6h`
  - 充電分断許容ギャップ `GAP_MAX = 5min`
  - IG-ONデバウンス `Δt_IGON_MERGE = 5min`
- 距離はハバースイン、座標/SOCはrolling median等で平滑化推奨

---

## 1. セッションの定義
**1セッション = 充電開始 → 充電終了 → 最初の ≥6h の放置（充電後初回のみ採用）**

- 充電後に現れる **6h未満の停車は無視**
- 放置終了は **IG-ON更新時刻（= 起動）** とする（固定）

---

## 2. 出力イベントと列（拡張）
各イベントで **時刻 / SOC / 緯度 / 経度** を保持。  
さらに **未来側アイドリング（IG-ONでの停車・起動後のアイドリング）** と **トータル放置区間** を列として追加。

### 2.1 基本イベント
- `charge_start_*`：充電開始
- `charge_end_*`：充電終了（癒着後の最後）
- `idling_start_*`：起動前（BWD）IG-ON停車開始
- `idling_end_*`：起動前（BWD）IG-ON停車終了
- `parking_start_*`：IG-OFF開始（未観測のため **`idling_end_*` と同一点**で近似）
- `parking_end_*`：**IG-ON更新時刻（= 放置終了）**

### 2.2 未来側アイドリング（起動後のIG-ON停車）
- `idling_future_start_*`：**IG-ON更新直後**から条件 `(距離≤DIST_TH or |ΔSOC|≤SOC_TH)` かつ `not is_charging` が続く最初の点（通常は `parking_end_*`）
- `idling_future_end_*`：上記条件が破綻する直前の点  
- `idling_future_duration_h`，`soc_drop_idling_future_%`，`max_disp_idling_future_m`

> 未来側は“放置”には含めない（**放置＝IG-OFF主体**）。ただし行動把握のために保持。

### 2.3 トータル放置区間（レポート用集約）
- `total_idle_block_start_ts`：`idling_start_ts`（起動前IG-ON停車の開始）
- `total_idle_block_end_ts`：`idling_future_end_ts`（起動後IG-ON停車の終了、無ければ `parking_end_ts`）
- `total_idle_block_duration_h`：上記の差
- 参考：`parking_duration_h`（= `parking_end_ts - parking_start_ts`）、`idling_duration_h`（起動前IG-ON停車）、`idling_future_duration_h`

---

## 3. 抽出ロジック（ハイブリッド：Cを外枠、A/Bで確定）

### 3.1 充電区間の癒着（前処理）
- 途切れた `is_charging` を **ギャップ≤`GAP_MAX` かつ 移動≤`DIST_TH` かつ SOCノイズ内** なら連結
- 一点ノイズ（1サンプルの反転）は前後に吸収
- 同一ステーション/半径≤`STATION_RADIUS` は連結を補強

### 3.2 IG-ON更新イベント列
- `tsu_igon_time` の更新行を抽出し、近接イベントは `Δt_IGON_MERGE` でデバウンス
- 各イベントをアンカー `anchor_k` として扱う

### 3.3 起動前アイドリング（過去方向 BWD）
- `anchor_k` の直前から過去へ、条件 **A or B** かつ `not is_charging` が続く限り拡張
  - **A**：`|ΔSOC| ≤ SOC_TH`（SOC安定）
  - **B**：`距離 ≤ DIST_TH`（移動なし）
- 最古の点が `idling_start`、`anchor_k-1` が `idling_end`
- `parking_start := idling_end`（IG-OFF開始の近似）
- `parking_end := anchor_k.ts`（**放置終了はIG-ON更新で固定**）

### 3.4 放置採用判定
- `parking_duration = parking_end - parking_start`
- `parking_duration ≥ PARK_TH` を満たす **最初の候補のみ** セッションとして採用
- 未満はスキップして次の IG-ONイベントへ

### 3.5 起動後アイドリング（未来方向 FWD：**保持のみ**）
- `anchor_k` から未来へ、条件 **A or B** かつ `not is_charging` が続く限り拡張
- 得られた区間を **未来側アイドリング** として列に保持
- **放置には含めない**（行動の文脈情報として活用）

---

## 4. 評価・例外
- **フェリー/牽引**：距離大でもSOC安定 → AでBWD継続（IG-OFF主体の駐車は `parking_*` で捉える）
- **都市峡谷のGPSノイズ**：距離大でもSOC安定 → Aで救済
- **エアコン待機**：距離小だがSOC減 → Bで救済（IG-ON停車）
- **充電混入**：BWD/FWDとも `is_charging=True` で強制終了
- **SOC再推定ジャンプ**：`parking_end` 近傍のSOCは参考値扱い（場所特定が主目的）

---

## 5. しきい値のチューニング
- 距離：IG-ON前後2点距離の分布95%点（ユーザー/エリア別）
- SOC：距離誤差内のSOC変化分布95%点（季節/時間帯別も可）
- 放置時間：放置長の分布＋**累積寄与（パレート図）**から、業務要件で決定

---
