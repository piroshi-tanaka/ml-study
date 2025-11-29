# ライブラリ解説: onnxruntime & skl2onnx

## 📋 目次

1. [概要](#概要)
2. [ONNX (Open Neural Network Exchange)](#onnx-open-neural-network-exchange)
3. [skl2onnx - scikit-learn to ONNX変換](#skl2onnx---scikit-learn-to-onnx変換)
4. [onnxruntime - ONNX推論エンジン](#onnxruntime---onnx推論エンジン)
5. [実践的な使い方](#実践的な使い方)
6. [よくある質問](#よくある質問)
7. [参考リンク](#参考リンク)

---

## 概要

### このドキュメントの目的

- `onnxruntime`と`skl2onnx`の役割を理解する
- 各ライブラリの使い方を習得する
- ONNX形式のメリットを理解する

### 関係図

```
┌─────────────────────────────────────────────────────────┐
│                   機械学習ワークフロー                    │
└─────────────────────────────────────────────────────────┘

【トレーニング】Python環境
   ↓
┌──────────────────┐
│  scikit-learn    │ モデルのトレーニング
│  (Python)        │
└────────┬─────────┘
         │
         │ skl2onnx で変換
         ↓
┌──────────────────┐
│  ONNX形式        │ 中間表現（プラットフォーム非依存）
│  (.onnx ファイル) │
└────────┬─────────┘
         │
         │ onnxruntime で推論
         ↓
┌─────────────────────────────────────────────┐
│              推論環境（複数選択可）            │
├─────────────────────────────────────────────┤
│  Python (onnxruntime)                       │
│  C++ (onnxruntime C++ API)                  │
│  Java, C#, JavaScript, ...                  │
│  組み込みLinux (AUTOSAR等)                   │
└─────────────────────────────────────────────┘
```

---

## ONNX (Open Neural Network Exchange)

### ONNXとは？

**ONNX (Open Neural Network Exchange)** は、機械学習モデルを表現するための**オープンな標準フォーマット**です。

#### 主な特徴

1. **フレームワーク間の互換性**
   - PyTorch、TensorFlow、scikit-learnなど、異なるフレームワークで学習したモデルを統一的に扱える

2. **プラットフォーム非依存**
   - Python、C++、Java、C#など、様々な言語/環境で使用可能

3. **最適化されたパフォーマンス**
   - 推論に特化した最適化が施される
   - ハードウェアアクセラレーション（GPU、専用チップ）のサポート

4. **エコシステム**
   - 多くの企業・コミュニティがサポート
   - 豊富なツールとライブラリ

### ONNXファイルの構造

```
ONNXモデル (.onnx)
├── グラフ (Graph)
│   ├── ノード (Nodes) - 演算子
│   ├── エッジ (Edges) - データフロー
│   └── 初期化子 (Initializers) - パラメータ
├── メタデータ
│   ├── バージョン情報
│   ├── プロデューサー情報
│   └── モデル説明
└── 入出力定義
    ├── 入力テンソルの形状・型
    └── 出力テンソルの形状・型
```

### なぜONNXを使うのか？

#### ユースケース1: クロスプラットフォーム展開

```
【開発環境】
Python + scikit-learn でモデルを開発
         ↓ ONNX変換
【本番環境】
- Webサーバー: Python (onnxruntime)
- モバイルアプリ: C++ (onnxruntime)
- 組み込み機器: C (ONNX Runtime for ARM)
```

#### ユースケース2: パフォーマンス最適化

- Pythonよりも高速な推論
- メモリ効率の向上
- バッチ推論の最適化

#### ユースケース3: デプロイの簡素化

- 依存関係の削減（scikit-learn全体が不要）
- モデルサイズの最適化
- バージョン管理の簡素化

---

## skl2onnx - scikit-learn to ONNX変換

### skl2onnxとは？

**skl2onnx** は、scikit-learnで学習したモデルをONNX形式に変換するPythonライブラリです。

- **公式GitHub**: https://github.com/onnx/sklearn-onnx
- **ライセンス**: MIT License
- **開発**: ONNX Community

### インストール

```bash
# pipの場合
pip install skl2onnx

# uvの場合
uv add skl2onnx

# condaの場合
conda install -c conda-forge skl2onnx
```

### 基本的な使い方

#### 1. シンプルな変換例

```python
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# モデルのトレーニング
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 入力の型を定義
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]

# ONNX変換
onnx_model = convert_sklearn(
    model,
    initial_types=initial_type,
    target_opset=12  # ONNXのバージョン
)

# ファイルに保存
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

#### 2. 回帰モデルの変換例

```python
from sklearn.linear_model import LinearRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# モデルのトレーニング
model = LinearRegression()
model.fit(X_train, y_train)

# ONNX変換
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# 保存
with open("linear_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

#### 3. パイプラインの変換

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# パイプラインの構築
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])
pipeline.fit(X_train, y_train)

# ONNX変換（パイプライン全体）
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(pipeline, initial_types=initial_type)
```

### 対応しているモデル

#### ✅ 分類器 (Classifiers)

- `LogisticRegression`
- `SVC` (Support Vector Classifier)
- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `DecisionTreeClassifier`
- `KNeighborsClassifier`
- `MLPClassifier`

#### ✅ 回帰 (Regressors)

- `LinearRegression`
- `Ridge`, `Lasso`
- `SVR` (Support Vector Regression)
- `RandomForestRegressor`
- `GradientBoostingRegressor`
- `DecisionTreeRegressor`
- `KNeighborsRegressor`
- `MLPRegressor`

#### ✅ 前処理 (Preprocessing)

- `StandardScaler`
- `MinMaxScaler`
- `RobustScaler`
- `LabelEncoder`
- `OneHotEncoder`
- `Normalizer`

#### ✅ その他

- `Pipeline`
- `ColumnTransformer`
- `PCA`
- `TruncatedSVD`

### パラメータ解説

#### `convert_sklearn()` の主要パラメータ

```python
onnx_model = convert_sklearn(
    model,                    # scikit-learnモデル
    initial_types=None,       # 入力の型定義（必須）
    target_opset=None,        # ONNXのバージョン
    options=None,             # 変換オプション
    white_op=None,            # 使用する演算子のホワイトリスト
    black_op=None,            # 使用しない演算子のブラックリスト
    final_types=None,         # 出力の型定義
    dtype=None,               # デフォルトのデータ型
    naming=None,              # ノードの命名規則
    model_optim=None,         # モデル最適化オプション
)
```

#### `initial_types` の定義

```python
from skl2onnx.common.data_types import (
    FloatTensorType,    # float32
    DoubleTensorType,   # float64
    Int64TensorType,    # int64
    StringTensorType,   # string
)

# 例1: シンプルな入力
initial_type = [('float_input', FloatTensorType([None, 10]))]
#                ↑名前          ↑型            ↑形状 [バッチ, 特徴量数]

# 例2: 複数入力
initial_type = [
    ('numeric_input', FloatTensorType([None, 5])),
    ('categorical_input', StringTensorType([None, 3]))
]

# 例3: 固定バッチサイズ
initial_type = [('float_input', FloatTensorType([32, 10]))]
```

#### `target_opset` について

ONNXのバージョンを指定します：

```python
# 推奨: 12以上
onnx_model = convert_sklearn(model, initial_types=..., target_opset=12)

# 最新: 15-18
onnx_model = convert_sklearn(model, initial_types=..., target_opset=18)
```

---

## onnxruntime - ONNX推論エンジン

### onnxruntimeとは？

**ONNX Runtime** は、ONNXモデルを高速に実行するための**クロスプラットフォーム推論エンジン**です。

- **公式サイト**: https://onnxruntime.ai/
- **GitHub**: https://github.com/microsoft/onnxruntime
- **開発**: Microsoft
- **ライセンス**: MIT License

### 主な特徴

1. **高速な推論**
   - CPU、GPU、専用ハードウェアで最適化
   - PyTorchやTensorFlowより高速な場合も

2. **クロスプラットフォーム**
   - Windows, Linux, macOS, iOS, Android
   - x86, ARM, WebAssembly

3. **多言語サポート**
   - Python, C++, C#, Java, JavaScript, Objective-C

4. **ハードウェアアクセラレーション**
   - CUDA (NVIDIA GPU)
   - TensorRT (NVIDIA)
   - OpenVINO (Intel)
   - CoreML (Apple)
   - DirectML (Windows)

### インストール

#### Python版

```bash
# CPU版（推奨）
pip install onnxruntime

# GPU版（CUDA対応）
pip install onnxruntime-gpu

# uvの場合
uv add onnxruntime
```

#### C++版

```bash
# Linuxの場合
wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
tar -xzf onnxruntime-linux-x64-1.18.0.tgz
sudo cp -r onnxruntime-linux-x64-1.18.0/include/* /usr/local/include/
sudo cp -r onnxruntime-linux-x64-1.18.0/lib/* /usr/local/lib/
sudo ldconfig
```

### Python APIの使い方

#### 1. 基本的な推論

```python
import onnxruntime as rt
import numpy as np

# セッションの作成
session = rt.InferenceSession("model.onnx")

# 入力・出力の名前を取得
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 推論データの準備
X_test = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

# 推論実行
result = session.run(
    [output_name],           # 出力名のリスト
    {input_name: X_test}     # 入力データの辞書
)

# 結果の取得
predictions = result[0]
print(predictions)
```

#### 2. バッチ推論

```python
# 複数サンプルを一度に推論
X_batch = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
], dtype=np.float32)

result = session.run([output_name], {input_name: X_batch})
predictions = result[0]  # shape: (3, num_outputs)
```

#### 3. セッションオプション

```python
import onnxruntime as rt

# セッションオプションの設定
options = rt.SessionOptions()
options.intra_op_num_threads = 4      # スレッド数
options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
options.enable_profiling = False      # プロファイリング

# セッションの作成
session = rt.InferenceSession("model.onnx", options)
```

#### 4. 実行プロバイダーの選択

```python
# CPU
session = rt.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])

# CUDA (GPU)
session = rt.InferenceSession("model.onnx", providers=['CUDAExecutionProvider'])

# 複数プロバイダー（優先順）
session = rt.InferenceSession(
    "model.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

### C++ APIの使い方

#### 1. 基本的な推論

```cpp
#include <onnxruntime_cxx_api.h>
#include <vector>

// 環境の初期化
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXInference");

// セッションオプション
Ort::SessionOptions session_options;

// セッションの作成
Ort::Session session(env, "model.onnx", session_options);

// 入力データの準備
std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
std::vector<int64_t> input_shape = {1, 3};

// メモリ情報
auto memory_info = Ort::MemoryInfo::CreateCpu(
    OrtArenaAllocator, OrtMemTypeDefault
);

// 入力テンソルの作成
Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    memory_info,
    input_data.data(),
    input_data.size(),
    input_shape.data(),
    input_shape.size()
);

// 推論実行
std::vector<const char*> input_names = {"float_input"};
std::vector<const char*> output_names = {"output"};

auto output_tensors = session.Run(
    Ort::RunOptions{nullptr},
    input_names.data(),
    &input_tensor,
    1,
    output_names.data(),
    1
);

// 結果の取得
float* output_data = output_tensors[0].GetTensorMutableData<float>();
```

### パフォーマンスチューニング

#### 1. グラフ最適化

```python
options = rt.SessionOptions()

# 最適化レベル
options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
# ORT_DISABLE_ALL: 最適化なし
# ORT_ENABLE_BASIC: 基本的な最適化
# ORT_ENABLE_EXTENDED: 拡張最適化
# ORT_ENABLE_ALL: すべての最適化
```

#### 2. スレッド数の調整

```python
options = rt.SessionOptions()
options.intra_op_num_threads = 4  # 演算内並列度
options.inter_op_num_threads = 2  # 演算間並列度
```

#### 3. メモリ最適化

```python
options = rt.SessionOptions()
options.enable_mem_pattern = True   # メモリパターンの最適化
options.enable_cpu_mem_arena = True # CPUメモリアリーナ
```

---

## 実践的な使い方

### パターン1: scikit-learn → ONNX → Python推論

```python
from sklearn.ensemble import RandomForestRegressor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
import numpy as np

# 1. モデルのトレーニング
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# 2. ONNX変換
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# 3. 保存
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# 4. ONNX推論
session = rt.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 5. 推論実行
X_test_float32 = X_test.astype(np.float32)
predictions = session.run([output_name], {input_name: X_test_float32})[0]

# 6. 精度検証
print(f"ONNX RMSE: {np.sqrt(mean_squared_error(y_test, predictions))}")
```

### パターン2: 前処理パイプライン込みの変換

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# パイプラインの構築
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier())
])
pipeline.fit(X_train, y_train)

# ONNX変換（前処理込み）
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(pipeline, initial_types=initial_type)

# 保存
with open("pipeline.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# 推論時は前処理不要！
session = rt.InferenceSession("pipeline.onnx")
# 生データをそのまま入力できる
predictions = session.run([output_name], {input_name: X_raw})
```

### パターン3: モデルの検証

```python
import onnxruntime as rt
from sklearn.metrics import accuracy_score

# scikit-learnとONNXの予測を比較
y_pred_sklearn = model.predict(X_test)

session = rt.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
y_pred_onnx = session.run([output_name], {input_name: X_test.astype(np.float32)})[0]

# 差分の確認
diff = np.abs(y_pred_sklearn - y_pred_onnx)
print(f"最大差分: {np.max(diff)}")
print(f"平均差分: {np.mean(diff)}")

# 精度の比較
print(f"scikit-learn精度: {accuracy_score(y_test, y_pred_sklearn)}")
print(f"ONNX精度: {accuracy_score(y_test, y_pred_onnx)}")
```

---

## よくある質問

### Q1: skl2onnxとonnxruntimeの違いは？

| ライブラリ | 役割 | 使用タイミング |
|-----------|------|--------------|
| **skl2onnx** | scikit-learn → ONNX変換 | モデルのトレーニング後 |
| **onnxruntime** | ONNX推論実行 | デプロイ時・推論時 |

### Q2: すべてのscikit-learnモデルが変換できる？

**A**: ほとんどのモデルが対応していますが、一部未対応もあります。

**対応状況の確認方法**:
```python
from skl2onnx import supported_converters
print(supported_converters())
```

### Q3: ONNX変換後、精度が変わる？

**A**: 基本的に変わりません。ただし、浮動小数点演算の実装差により、微小な差（1e-6程度）が生じる場合があります。

### Q4: ONNXモデルのサイズは？

**A**: scikit-learnの`.pkl`ファイルと同程度、または少し小さくなることが多いです。

### Q5: onnxruntimeの方が速い？

**A**: 一般的には、onnxruntimeの方が高速です。特に：
- バッチ推論
- CPU推論
- 最適化が有効な場合

ベンチマーク例：
```python
import time

# scikit-learn
start = time.time()
pred_sklearn = model.predict(X_test)
print(f"scikit-learn: {time.time() - start:.4f}秒")

# onnxruntime
start = time.time()
pred_onnx = session.run([output_name], {input_name: X_test})[0]
print(f"onnxruntime: {time.time() - start:.4f}秒")
```

### Q6: ONNXモデルのバージョン管理は？

**A**: `.onnx`ファイルをGitで管理するか、モデルレジストリ（MLflow等）を使用します。

### Q7: エラー「Unsupported model」が出る

**A**: そのモデルがskl2onnxで未対応の可能性があります。

**解決策**:
1. skl2onnxを最新版にアップデート
2. サポート状況を確認
3. カスタム変換器を作成

### Q8: C++版とPython版で結果が違う

**A**: 浮動小数点演算の差です。通常は1e-6程度の差なので問題ありません。

**確認方法**:
```python
# Python側
predictions_python = session.run([output_name], {input_name: X_test})[0]

# C++側の結果と比較
diff = np.abs(predictions_python - predictions_cpp)
print(f"最大差分: {np.max(diff)}")  # 1e-6以下なら問題なし
```

---

## トラブルシューティング

### エラー1: "ONNX Runtime failed to initialize"

**原因**: ライブラリのインストール不備

**解決策**:
```bash
pip uninstall onnxruntime
pip install onnxruntime --no-cache-dir
```

### エラー2: "Cannot find implementation for operator"

**原因**: ONNXのバージョン（opset）が合っていない

**解決策**:
```python
# target_opsetを指定
onnx_model = convert_sklearn(model, initial_types=..., target_opset=12)
```

### エラー3: "Input type not supported"

**原因**: 入力データ型が正しくない

**解決策**:
```python
# float32に変換
X_test = X_test.astype(np.float32)
predictions = session.run([output_name], {input_name: X_test})
```

### エラー4: "Shape mismatch"

**原因**: 入力の形状が定義と異なる

**解決策**:
```python
# 形状を確認
print("入力形状（定義）:", session.get_inputs()[0].shape)
print("入力形状（実際）:", X_test.shape)

# 形状を合わせる
X_test = X_test.reshape(-1, num_features)
```

---

## 参考リンク

### 公式ドキュメント

- [ONNX公式サイト](https://onnx.ai/)
- [ONNX Runtime公式サイト](https://onnxruntime.ai/)
- [skl2onnx GitHub](https://github.com/onnx/sklearn-onnx)
- [skl2onnxドキュメント](http://onnx.ai/sklearn-onnx/)

### チュートリアル

- [ONNX Tutorials](https://github.com/onnx/tutorials)
- [ONNX Runtime Examples](https://github.com/microsoft/onnxruntime/tree/main/samples)

### API リファレンス

- [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/)
- [ONNX Runtime C++ API](https://onnxruntime.ai/docs/api/c/)

### コミュニティ

- [ONNX GitHub Discussions](https://github.com/onnx/onnx/discussions)
- [ONNX Runtime GitHub Issues](https://github.com/microsoft/onnxruntime/issues)

---

## まとめ

### skl2onnx

- ✅ scikit-learn → ONNX変換ツール
- ✅ ほとんどのscikit-learnモデルに対応
- ✅ パイプライン全体を変換可能
- ✅ 簡単なAPIで使いやすい

### onnxruntime

- ✅ ONNX推論エンジン
- ✅ 高速な推論（CPUでもGPUでも）
- ✅ クロスプラットフォーム（Python、C++、等）
- ✅ 最適化機能が豊富

### 使い分け

| タイミング | ライブラリ | 目的 |
|-----------|-----------|------|
| **開発時** | skl2onnx | モデル変換 |
| **検証時** | onnxruntime (Python) | 精度確認 |
| **本番時** | onnxruntime (Python/C++) | 推論実行 |

### 次のステップ

1. **実際に試す**
   - [01. PythonからONNX形式を作成する手順](./01_Python_ONNX作成手順.md)

2. **C++で推論**
   - [03. C++推論実行手順](./03_C++推論実行手順.md)

3. **パフォーマンス最適化**
   - グラフ最適化
   - ハードウェアアクセラレーション

---

**作成日**: 2025-11-02  
**バージョン**: 1.0  
**対象**: ONNX初心者〜中級者

