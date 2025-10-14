# EVバッテリ劣化抑制_放置予測 — 要件書（最終/簡潔版）


---

## 0. 目的
- 充電イベントの **直後に最初に発生する長時間放置（≥6h）** を厳密にリンク化し、  
  **充電開始の時間帯×充電クラスタ → 放置開始の時間帯×放置クラスタ** の関係を可視化・定量化。
- さらに **1日単位の遷移** と **充電あり日 vs なし日** を比較し、  
  充電が日常ルートへの“挿入”か“独立行動”かを判定する。

---

## 1. スコープ/前提
- すべて **hashvin単位** で前処理・集計・可視化を行う。
- 入力はセッション単位（滞在 `inactive` / 充電 `charging`）。移動中・停滞充電は除外済み。
- タイムゾーン: Asia/Tokyo、長時間放置: `duration_minutes ≥ 360`。

---

## 2. 入力列
- `hashvin, session_cluster, session_type(inactive/charging), start_time, end_time, duration_minutes, start_soc, end_soc, change_soc, start_lat, start_lon, end_lat, end_lon`

---

## 3. 派生列（hashvinごと）
- `weekday = start_time.dayofweek`（0=Mon…6=Sun）
- `start_hour = start_time.hour`
- `date = start_time.date`
- `is_long_park = (session_type=='inactive') & (duration_minutes>=360)`
- `prev_* / next_*`（シフトで直前・直後の種別/クラスタ/時間を参照：可視化補助）

---

## 4. 充電→長時間放置リンク（確定ルール）
- 各充電イベント **c** の終了から **次の充電イベント開始まで** の区間内にある  
  **最初の長時間放置（`is_long_park==True`）** をリンク（**1件のみ**）。
- 区間内に長時間放置が **存在しない** 場合はリンク **なし** とする。
- リンクテーブル列（例）  
  `hashvin, weekday, charge_cluster, charge_start_time, charge_start_hour, charge_end_time, park_cluster(NA可), park_start_time(NA), park_start_hour(NA), park_duration_minutes(NA), gap_minutes(NA), dist_charge_to_park_km(任意)`

---

## 5. ミックス集計（充電×放置の“対”）
### 5.1 集約定義
- **集約版（充電クラスタ無指定）**  
  `count(h_c, h_p, c_p)` … 充電開始時刻 `h_c`、放置開始時刻 `h_p`、放置クラスタ `c_p`
- **充電クラスタ別**  
  `count(c_c, h_c, h_p, c_p)` … 充電クラスタ `c_c` を条件に加える

### 5.2 条件付き確率（平滑化推奨）
- `P(c_p | h_c, h_p)`、`P(c_p | c_c, h_c, h_p)` を列正規化（疎セルは閾値でNA、+α平滑化）。

---

## 6. 日内遷移（1日区切り・時間区間の“存在判定”）
- **スロット化は行わない**。各 `inactive` の **区間 [start_time, end_time)** が  
  時間帯ビン（例 `0–6, 6–9, 9–12, 12–15, 15–18, 18–21, 21–24`）に **重なっているか（存在判定）** を用いる。
- 日ごとにビン順で「代表クラスタ」（該当区間に**存在したクラスタ**のうち、時間重複量が最大のもの等）を決め、  
  `(time_bin, cluster)` 列の連鎖として **日内遷移** を構築。
- エッジは **隣接ビン間** の `(bin_i, cluster_i) → (bin_j, cluster_j)` をカウント。

---

## 7. 充電あり日 vs なし日（比較設計）
- `has_charge_day(date)` を作成（その日のセッションに charging が1つでもあれば True）。
- 比較対象：
  - **日内クラスタ分布**（hour×cluster の存在比）  
  - **日内遷移行列 T**（クラスタ間の隣接遷移確率）
- 指標：
  - **JS距離**（分布差）  
  - **ΔT = T_charge − T_nocharge**（遷移差のヒートマップ）  
  - **route_return_ratio**（充電後 N 時間以内に“充電前の主クラスタ”へ戻る確率）  
  - **charge_specific_ratio**（充電日の「charging→近距離短時間park→移動」割合）

---

## 8. 可視化（hashvinごと）
### 8.1 ミックス（充電×放置）
- **H2Dヒートマップ（集約版）**  
  - 軸: y=充電開始 `h_c`、x=放置開始 `h_p`  
  - 値: Top1放置クラスタ確率 or エントロピー  
  - 目的: 「この時間に充電 → 何時からどの放置先？」を俯瞰
- **H2Dヒートマップ（充電クラスタ別）**  
  - 小倍数（facet）で `charge_cluster` ごと表示  
  - 目的: ステーション固有の偏りを把握
- **条件付き棒グラフ（充電クラスタ別）**  
  - x=`h_c`、y=`P(next_cluster | c_c, h_c, h_p_bin)`、hue=`next_cluster`、facet=`h_p_bin`  
  - 目的: 実務で使う「時間帯×拠点→放置先」地図

### 8.2 日内遷移（存在判定ビン）
- **Sankey / 時系列ネットワーク**  
  - ノード=`(time_bin, cluster)`、リンク幅=件数  
  - 目的: 「朝A→午前B→夕C→夜A」のループや分岐
- **Timeline Heatmap**  
  - y=cluster、x=hour、値=**存在割合**（区間重なりがあれば1として集計）  
  - 目的: 各クラスタの“活動時間帯”を俯瞰

### 8.3 充電あり日 vs なし日
- **分布差分ヒートマップ**（hour×cluster の存在割合：charge − nocharge）
- **遷移行列差分ヒートマップ**（ΔT）
- **指標棒グラフ**（route_return_ratio / js_distance / charge_specific_ratio）

---

## 9. この設計で傾向が分かる理由
- **充電→最初の長時間放置**のみをリンクし、**次の充電まで**を上限とすることで、  
  「充電が引き起こす直後の主要放置」をノイズなく抽出できる（因果連鎖に近い）。
- **時間×場所の条件付き確率**で見るため、  
  「この時間にこの拠点で充電 → この時間からこの放置先」という **予測構造** が可視化される。
- **スロット化せず区間重なりの“存在判定”**を用いることで、  
  時間帯ごとの **実在性** に基づく日内遷移を歪み少なく表現できる。
- **充電あり/なしの差分**と **復帰率指標** により、  
  充電が日常ルートの **挿入イベント** か **独立行動** かを定量的に切り分けられる。

---

## 10. 受け入れ基準（AC）
- [ ] 充電ごとに、「次の充電まで」の間で **最初の長時間放置のみ** をリンクし、該当なしは **リンクなし** を表現できている。  
- [ ] `P(c_p | h_c, h_p)`（集約）と `P(c_p | c_c, h_c, h_p)`（充電クラスタ別）が算出され、疎セルは適切に扱う。  
- [ ] H2Dヒートマップ（集約/拠点別）と条件付き棒が意図通り描画される。  
- [ ] 日内遷移（Sankey/Network）と Timeline Heatmap が「存在判定」に基づき描画される。  
- [ ] 充電あり日 vs なし日で、分布・遷移の差分と指標（復帰率/JS距離/特化比率）が比較できる。
