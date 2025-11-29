# C++でのONNX推論実装ガイド

## 概要

このディレクトリには、WSL Ubuntu環境でC++からONNXモデルを使用して時系列予測を行うためのコードが含まれています。

## 目的

- AUTOSAR類似環境（Linux）でのML推論の検証
- Python推論との精度比較
- C++実装での推論性能の評価

## ファイル構成

```
cpp_inference/
├── CMakeLists.txt              # CMakeビルド設定
├── onnx_inference.cpp          # C++推論コード
├── README.md                   # このファイル
├── time_series_model.onnx      # ONNXモデル
├── test_data.csv               # テストデータ（特徴量）
└── test_labels.csv             # 正解ラベル＋Python予測結果
```

## セットアップ手順（WSL Ubuntu）

### ステップ1: WSL Ubuntuへのアクセス

PowerShellまたはコマンドプロンプトから：

```bash
wsl
```

### ステップ2: 必要なパッケージのインストール

```bash
# システムパッケージの更新
sudo apt update
sudo apt upgrade -y

# ビルドツールのインストール
sudo apt install -y build-essential cmake wget unzip
```

### ステップ3: ONNX Runtime C++ APIのインストール

#### 方法A: 公式ビルド済みバイナリを使用（推奨）

```bash
# バージョンを指定（最新版は公式サイトで確認）
ONNX_VERSION="1.18.0"

# ダウンロード
cd ~
wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz

# 解凍
tar -xzf onnxruntime-linux-x64-${ONNX_VERSION}.tgz

# システムディレクトリにコピー
sudo cp -r onnxruntime-linux-x64-${ONNX_VERSION}/include/* /usr/local/include/
sudo cp -r onnxruntime-linux-x64-${ONNX_VERSION}/lib/* /usr/local/lib/

# ライブラリパスの更新
sudo ldconfig
```

#### 方法B: apt経由でインストール（簡単だが古いバージョンの可能性）

```bash
# ONNX Runtimeパッケージの検索
apt search onnxruntime

# インストール（利用可能な場合）
sudo apt install -y libonnxruntime-dev
```

### ステップ4: プロジェクトディレクトリへの移動

WindowsファイルシステムのパスをWSL形式に変換：

```bash
# Cドライブの場合
cd /mnt/c/workspace/src/ml-study/scikit_learn_onnx_repo/cpp_inference

# 確認
ls -la
```

### ステップ5: ビルド

```bash
# ビルドディレクトリの作成
mkdir build
cd build

# CMakeの実行
cmake ..

# ビルド
make

# 確認
ls -l onnx_inference
```

### ステップ6: 実行

```bash
# ビルドディレクトリから実行
cd /mnt/c/workspace/src/ml-study/scikit_learn_onnx_repo/cpp_inference/build
./onnx_inference ../time_series_model.onnx ../test_data.csv ../test_labels.csv

# または、cpp_inferenceディレクトリから実行
cd ..
./build/onnx_inference
```

## 期待される出力

```
========================================
  ONNX時系列予測推論（C++版）
========================================

📂 ファイル読み込み中...
  ONNXモデル: time_series_model.onnx
  テストデータ: test_data.csv
  正解ラベル: test_labels.csv

✓ データ読み込み完了
  サンプル数: 28
  特徴量数: 12

🔧 ONNX Runtimeの初期化中...
  入力ノード数: 1
  出力ノード数: 1
  入力名: float_input
  出力名: variable

🚀 推論実行中...
  進捗: 10/28 サンプル
  進捗: 20/28 サンプル
  進捗: 28/28 サンプル

✓ 推論完了

============================================================
【精度評価】
============================================================

【C++ ONNX推論】
  RMSE: XX.XXXXXX
  MAE:  XX.XXXXXX
  R²:   X.XXXXXX

【Python ONNX vs C++ ONNX】
  最大差分: X.XXXXXXe-XX
  平均差分: X.XXXXXXe-XX
  ✓ Python ONNXとC++ ONNXの予測はほぼ一致！

============================================================
【予測結果サンプル（最初の5件）】
============================================================
No.  実績値    Python    C++予測   誤差
------------------------------------------------------------
  1   504.00   315.42   315.42    188.58
  2   404.00   313.51   313.51     90.49
  3   359.00   310.35   310.35     48.65
  4   310.00   307.91   307.91      2.09
  5   337.00   308.76   308.76     28.24

============================================================
✓ すべての処理が完了しました！
============================================================
```

## トラブルシューティング

### エラー1: `onnxruntime_cxx_api.h` が見つからない

**原因**: ONNX Runtimeが正しくインストールされていない

**解決策**:
```bash
# インクルードパスの確認
ls /usr/local/include/onnxruntime

# なければ再インストール
sudo cp -r ~/onnxruntime-linux-x64-*/include/* /usr/local/include/
```

### エラー2: `libonnxruntime.so` が見つからない

**原因**: ライブラリパスが通っていない

**解決策**:
```bash
# ライブラリの確認
ls /usr/local/lib/libonnxruntime*

# ライブラリパスの更新
sudo ldconfig

# それでもダメな場合、環境変数を設定
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

### エラー3: CMakeでONNX Runtimeが見つからない

**解決策**:
```bash
# CMakeに明示的にパスを指定
cmake -DONNXRUNTIME_ROOTDIR=/usr/local ..
```

### エラー4: 実行時に "cannot open shared object file"

**解決策**:
```bash
# 実行時にライブラリパスを指定
LD_LIBRARY_PATH=/usr/local/lib ./onnx_inference
```

## パフォーマンス測定

推論時間を測定したい場合、コードを以下のように修正できます：

```cpp
#include <chrono>

// 推論前
auto start = std::chrono::high_resolution_clock::now();

// 推論実行
// ...

// 推論後
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
std::cout << "推論時間: " << duration.count() / 1000.0 << " ms" << std::endl;
```

## AUTOSAR環境への展開

現在のWSL Ubuntu環境での検証が成功したら、以下のステップでAUTOSAR環境に展開できます：

1. **クロスコンパイル環境の構築**
   - AUTOSARターゲット用のツールチェーンを設定
   - ONNX Runtimeをターゲット環境用にビルド

2. **メモリ制約への対応**
   - モデルサイズの最適化
   - 静的メモリ割り当ての検討

3. **リアルタイム性の検証**
   - 推論時間の測定
   - 最悪実行時間（WCET）の評価

4. **安全性の考慮**
   - エラーハンドリングの強化
   - フェイルセーフ機構の実装

## 次のステップ

- ✅ WSL Ubuntu環境でのC++推論の動作確認
- ✅ Python推論との精度比較
- ⏭ 推論時間の測定と最適化
- ⏭ AUTOSAR環境への移植検討

## 参考リンク

- [ONNX Runtime C++ API Documentation](https://onnxruntime.ai/docs/api/c/)
- [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases)
- [AUTOSAR Adaptive Platform](https://www.autosar.org/)

