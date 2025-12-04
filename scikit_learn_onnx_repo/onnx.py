# ============================================
# 必要ライブラリ
# ============================================
# 初回だけ:
#   pip install scikit-learn pandas numpy onnx skl2onnx onnxruntime

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

# ============================================
# 1. ダミーデータ生成
#    - hashvin: ユーザーID相当のカテゴリ
#    - timestamp: 時刻
# ============================================

rng = np.random.default_rng(42)

n_hashvin = 200    # ユーザー数（高カーディナリティ想定）
n_days = 14        # 日数
freq_min = 60      # 60分間隔

start_dt = datetime(2025, 1, 1, 0, 0, 0)

timestamps = []
hashvins = []

for vin_idx in range(n_hashvin):
    vin_id = f"HASHVIN_{vin_idx:05d}"
    for i in range(int(24 * n_days * 60 / freq_min)):
        ts = start_dt + timedelta(minutes=freq_min * i)
        timestamps.append(ts)
        hashvins.append(vin_id)

df = pd.DataFrame({
    "hashvin": hashvins,
    "timestamp": timestamps,
})

# ============================================
# 2. 基本日時特徴量
#    - year はそのまま数値で利用
#    - month/day/hour は後で sin/cos にする
# ============================================

df["year"] = df["timestamp"].dt.year.astype(np.int32)
df["month"] = df["timestamp"].dt.month.astype(np.int32)
df["day"] = df["timestamp"].dt.day.astype(np.int32)
df["hour"] = df["timestamp"].dt.hour.astype(np.int32)

# ============================================
# 3. month/day/hour の sin/cos（double で計算 → float32 へキャスト）
# ============================================

# double (float64) で正規化
df["month_norm"] = (df["month"].astype(np.float64) - 1.0) / np.float64(12.0)
df["day_norm"]   = (df["day"].astype(np.float64)   - 1.0) / np.float64(31.0)
df["hour_norm"]  =  df["hour"].astype(np.float64)        / np.float64(24.0)

# sin/cos を float64 で計算
for col in ["month_norm", "day_norm", "hour_norm"]:
    df[f"{col}_sin64"] = np.sin(2.0 * np.pi * df[col])
    df[f"{col}_cos64"] = np.cos(2.0 * np.pi * df[col])

# float64 → float32 にキャスト（モデルに渡す用）
time_cols_64 = [
    "month_norm_sin64", "month_norm_cos64",
    "day_norm_sin64",   "day_norm_cos64",
    "hour_norm_sin64",  "hour_norm_cos64",
]

for col64 in time_cols_64:
    col32 = col64.replace("64", "32")
    df[col32] = df[col64].astype(np.float32)

# 誤差チェック（オマケ：double→float32 の誤差感）
for col64 in time_cols_64:
    col32 = col64.replace("64", "32")
    diff = np.abs(df[col64] - df[col32].astype(np.float64))
    print(
        f"{col64} vs {col32} | "
        f"max diff = {diff.max():.3e}, mean diff = {diff.mean():.3e}"
    )

# year もモデル用には float32 に揃える
df["year_f32"] = df["year"].astype(np.float32)

# モデルに渡す時間系特徴量（全部 float32）
feature_cols_time = [
    "year_f32",
    "month_norm_sin32", "month_norm_cos32",
    "day_norm_sin32",   "day_norm_cos32",
    "hour_norm_sin32",  "hour_norm_cos32",
]

# ============================================
# 4. ダミー目的変数 target（SOC 風）
#    - hashvin ごとのオフセット + 日周性 + ノイズ
# ============================================

hashvin_codes = df["hashvin"].astype("category").cat.codes.to_numpy()
daily_phase = np.sin(2 * np.pi * df["hour_norm"])

y = (
    50.0
    + hashvin_codes * 0.01            # ユーザーごとの差
    + daily_phase * 10.0              # 日周パターン
    + rng.normal(0, 2.0, size=len(df))  # ノイズ
).astype(np.float32)

df["target"] = y

# ============================================
# 5. 時系列で train/test 分割（最後の 20% を test）
# ============================================

df = df.sort_values("timestamp").reset_index(drop=True)
n_total = len(df)
split_idx = int(n_total * 0.8)

df_train = df.iloc[:split_idx].copy()
df_test = df.iloc[split_idx:].copy()

# ============================================
# 6. hashvin のターゲットエンコーディング（mean encoding）
#    - train で hashvin ごとの平均 target
#    - test はそのテーブルを参照。未知 hashvin は全体平均にフォールバック
# ============================================

te_table = (
    df_train
    .groupby("hashvin")["target"]
    .mean()
    .astype(np.float32)
    .rename("hashvin_te")
    .reset_index()
)

df_train = df_train.merge(te_table, on="hashvin", how="left")
df_test = df_test.merge(te_table, on="hashvin", how="left")

global_mean = df_train["target"].mean().astype(np.float32)
df_train["hashvin_te"].fillna(global_mean, inplace=True)
df_test["hashvin_te"].fillna(global_mean, inplace=True)

# ※ te_table と global_mean は ECU 側の前処理でも必要な情報
#   （hashvin -> hashvin_te のマッピング＋全体平均）

# ============================================
# 7. 最終的な特徴量行列 X / 目的変数 y
#    - すべて float32
# ============================================

feature_cols_all = feature_cols_time + ["hashvin_te"]

X_train = df_train[feature_cols_all].astype(np.float32).to_numpy()
y_train = df_train["target"].astype(np.float32).to_numpy()

X_test = df_test[feature_cols_all].astype(np.float32).to_numpy()
y_test = df_test["target"].astype(np.float32).to_numpy()

print("X_train shape:", X_train.shape, "dtype:", X_train.dtype)
print("X_test shape :", X_test.shape,  "dtype:", X_test.dtype)

# ============================================
# 8. HistGradientBoostingRegressor モデル定義＆学習
# ============================================

reg = HistGradientBoostingRegressor(
    max_depth=6,
    learning_rate=0.1,
    max_iter=200,
    random_state=42,
)
reg.fit(X_train, y_train)

y_pred_sklearn = reg.predict(X_test).astype(np.float32)
mae_sklearn = mean_absolute_error(y_test, y_pred_sklearn)
print("MAE (sklearn):", mae_sklearn)

# ============================================
# 9. ONNX へ変換（skl2onnx）
# ============================================

n_features = X_train.shape[1]
initial_type = [("input", FloatTensorType([None, n_features]))]

onnx_model = convert_sklearn(
    reg,
    initial_types=initial_type,
    target_opset=15,   # 環境に応じて変更可
)

onnx_path = "hgb_timeseries_hashvin_te_float32.onnx"
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Saved ONNX model to:", onnx_path)

onnx.checker.check_model(onnx_model)
print("ONNX model checked OK.")

# ============================================
# 10. ONNX Runtime (CPUExecutionProvider, ECU 想定) で推論
# ============================================

sess = ort.InferenceSession(
    onnx_path,
    providers=["CPUExecutionProvider"],  # Ubuntu ECU 前提なら基本これ
)

input_name = sess.get_inputs()[0].name
print("ONNX input name:", input_name)

# ONNX 推論
y_pred_onnx = sess.run(
    None,
    {input_name: X_test.astype(np.float32)}
)[0].astype(np.float32)

mae_onnx = mean_absolute_error(y_test, y_pred_onnx)
print("MAE (ONNX):", mae_onnx)

# ============================================
# 11. sklearn vs ONNX の予測差（double→float32 の影響も含めた最終確認）
# ============================================

abs_diff = np.abs(y_pred_sklearn - y_pred_onnx)
print("sklearn vs ONNX 予測差の統計")
print("  max abs diff:", abs_diff.max())
print("  mean abs diff:", abs_diff.mean())
print("  先頭10件 sklearn:", y_pred_sklearn[:10])
print("  先頭10件 ONNX   :", y_pred_onnx[:10])