# scikit-learn → ONNX → C++推論 完全ガイド

## 📚 ドキュメント一覧

このディレクトリには、scikit-learnモデルをONNX形式に変換し、WSL Ubuntu環境でC++から推論を実行するまでの完全な手順書が含まれています。

---

## 🎯 プロジェクトの目的

**AUTOSAR環境での機械学習推論の実現可能性を検証する**

- Python（scikit-learn）で学習したモデルをONNX形式に変換
- Linux環境（WSL Ubuntu）でC++から推論を実行
- Python推論との精度を比較し、AUTOSAR環境での実現可能性を評価

---

## 📖 ドキュメントの構成

### [01. PythonからONNX形式を作成する手順](./01_Python_ONNX作成手順.md)

**所要時間**: 20-30分（初回）

**内容**:
- Jupyter Notebookでの環境構築
- 時系列予測モデルのトレーニング
- scikit-learn → ONNX変換
- テストデータのエクスポート

**成果物**:
- `time_series_model.onnx` - ONNXモデル
- `test_data.csv` - テストデータ
- `test_labels.csv` - 正解ラベル

---

### [02. WSL Ubuntu環境構築手順](./02_WSL_Ubuntu_環境構築手順.md)

**所要時間**: 30-40分（初回）

**内容**:
- WSLとUbuntuのインストール
- Ubuntu初期設定
- ONNX Runtime C++のインストール
- VS Code + WSL拡張機能の設定

**成果物**:
- WSL Ubuntu 22.04 LTS環境
- ONNX Runtime C++ API (v1.18.0)
- ビルドツール（gcc, g++, cmake）

---

### [03. C++推論実行手順](./03_C++推論実行手順.md)

**所要時間**: 5-10分

**内容**:
- C++プロジェクトのビルド
- ONNX推論の実行
- Python推論との精度比較
- 結果の検証と解釈

**成果物**:
- C++実行ファイル
- 推論結果（精度評価）
- AUTOSAR環境への実現可能性の知見

---

### [ライブラリ解説: onnxruntime & skl2onnx](./ライブラリ解説_onnxruntime_skl2onnx.md) ⭐ 参考資料

**所要時間**: 読み物（15-20分）

**内容**:
- ONNXの基本概念
- skl2onnxの詳細な使い方
- onnxruntimeの機能とAPI
- 実践的なコード例
- よくある質問とトラブルシューティング

**対象**:
- ライブラリを深く理解したい方
- カスタマイズや応用を検討している方
- トラブルシューティングの参考に

---

## 🚀 クイックスタート

### 1. Python側（Windows）

```bash
# プロジェクトディレクトリで
cd C:\workspace\src\ml-study

# パッケージインストール
uv sync

# Jupyter Notebook起動
uv run jupyter notebook

# time_series_onnx_demo.ipynb を開いて全セルを実行
```

### 2. WSL Ubuntu側（Linux）

```bash
# WSL起動
wsl

# プロジェクトディレクトリへ移動
cd /mnt/c/workspace/src/ml-study/scikit_learn_onnx_repo/cpp_inference

# 環境構築（初回のみ）
chmod +x setup_and_build.sh
./setup_and_build.sh

# ビルドと実行
rm -rf build
mkdir build
cd build
cmake ..
make
cd ..
./run_inference.sh
```

---

## 📊 プロジェクト構造

```
ml-study/
├── pyproject.toml                          # Python依存関係
└── scikit_learn_onnx_repo/
    ├── time_series_onnx_demo.ipynb         # メインNotebook
    ├── time_series_model.onnx              # ONNXモデル
    ├── *.png (4つ)                         # 可視化結果
    │
    ├── docs/                               # ドキュメント ★
    │   ├── README.md                       # このファイル
    │   ├── 01_Python_ONNX作成手順.md
    │   ├── 02_WSL_Ubuntu_環境構築手順.md
    │   ├── 03_C++推論実行手順.md
    │   └── ライブラリ解説_onnxruntime_skl2onnx.md  # 参考資料
    │
    └── cpp_inference/                      # C++推論 ★
        ├── CMakeLists.txt                  # ビルド設定
        ├── onnx_inference.cpp              # C++推論コード
        ├── time_series_model.onnx          # ONNXモデル（コピー）
        ├── test_data.csv                   # テストデータ
        ├── test_labels.csv                 # 正解ラベル
        ├── setup_and_build.sh              # セットアップスクリプト
        ├── run_inference.sh                # 実行スクリプト
        ├── README.md                       # 詳細ガイド
        ├── QUICKSTART.md                   # クイックスタート
        └── build/                          # ビルド成果物
            └── onnx_inference              # 実行ファイル
```

---

## ✅ 動作確認済み環境

| 項目 | バージョン |
|------|-----------|
| **OS** | Windows 11 |
| **WSL** | WSL2 (2.0.14.0) |
| **Ubuntu** | Ubuntu 22.04.3 LTS |
| **Python** | 3.12+ |
| **gcc/g++** | 11.4.0 |
| **CMake** | 3.22.1 |
| **ONNX Runtime** | 1.18.0 |

---

## 🎯 期待される結果

### 精度の一致

```
【Python ONNX vs C++ ONNX】
  最大差分: 1.234567e-05  ← 1e-05以下
  平均差分: 3.456789e-06  ← 1e-06程度
  ✓ Python ONNXとC++ ONNXの予測はほぼ一致！
```

### 推論の成功

- テストデータ27サンプルすべてで推論成功
- RMSE、MAE、R²がPythonとほぼ同じ
- エラーなく完了

---

## 💡 よくある質問

### Q1: WSL/Ubuntuを使ったことがない

A: [02. WSL Ubuntu環境構築手順](./02_WSL_Ubuntu_環境構築手順.md) で初心者向けに詳しく説明しています。

### Q2: C++のコードを書いたことがない

A: コードは既に用意されています。ビルドと実行のコマンドをコピー＆ペーストするだけでOKです。

### Q3: エラーが出て進まない

A: 各ドキュメントの「トラブルシューティング」セクションを参照してください。それでも解決しない場合は、エラーメッセージを記録して確認してください。

### Q4: どのドキュメントから始めればいい？

A: [01. PythonからONNX形式を作成する手順](./01_Python_ONNX作成手順.md) から順番に進めてください。

### Q5: 所要時間はどのくらい？

A: 
- 初回（すべて新規）: 約1.5-2時間
- WSL/Ubuntu設定済み: 約30-40分
- データ・環境すべて済み: 約5-10分

---

## 🔧 トラブルシューティング

### 全般

1. **各ステップを順番に実行**
   - 飛ばさずに順番に進める
   - 前のステップが完了してから次へ

2. **エラーメッセージを確認**
   - エラーの内容をよく読む
   - 各ドキュメントのトラブルシューティングを参照

3. **環境の確認**
   - ファイルの存在を確認
   - パスが正しいか確認
   - バージョンが合っているか確認

### よくあるエラー

| エラー | 原因 | 解決策 |
|-------|------|--------|
| ファイルが見つからない | パスが違う | `pwd`で現在地を確認 |
| Permission denied | 実行権限がない | `chmod +x *.sh` |
| ヘッダーが見つからない | ONNX Runtime未インストール | 02のステップ6を再実行 |
| ライブラリが見つからない | パスが通っていない | `sudo ldconfig` |

---

## 📝 チェックリスト

### 01. Python編

- [ ] `uv sync` が成功
- [ ] Jupyter Notebookが起動
- [ ] すべてのセルがエラーなく実行
- [ ] `cpp_inference/` ディレクトリが作成
- [ ] ONNXモデルとCSVファイルが生成

### 02. 環境構築編

- [ ] WSLがインストール済み
- [ ] Ubuntu 22.04が起動
- [ ] システムがアップデート済み
- [ ] ONNX Runtimeがインストール済み
- [ ] VS Code + WSL拡張機能が動作

### 03. C++推論編

- [ ] CMakeがエラーなく完了
- [ ] makeでビルド成功
- [ ] 推論が実行できる
- [ ] Python vs C++の差分が小さい
- [ ] すべての処理が完了

---

## 🎓 学習リソース

### プロジェクト内ドキュメント

- [ライブラリ解説: onnxruntime & skl2onnx](./ライブラリ解説_onnxruntime_skl2onnx.md) ⭐ 推奨
  - ONNXの基本概念
  - ライブラリの詳細な使い方
  - 実践的なコード例

### ONNX関連

- [ONNX公式サイト](https://onnx.ai/)
- [ONNX Runtime公式ドキュメント](https://onnxruntime.ai/docs/)
- [skl2onnx GitHub](https://github.com/onnx/sklearn-onnx)

### WSL/Ubuntu

- [WSL公式ドキュメント](https://learn.microsoft.com/ja-jp/windows/wsl/)
- [Ubuntu公式サイト](https://ubuntu.com/)

### AUTOSAR

- [AUTOSAR公式サイト](https://www.autosar.org/)
- [AUTOSAR Adaptive Platform](https://www.autosar.org/standards/adaptive-platform/)

---

## 📞 サポート

### ドキュメントの改善

このドキュメントで不明な点や改善点がありましたら、フィードバックをお願いします。

### バグ報告

実行中にエラーが発生した場合は、以下の情報を記録してください：
- エラーメッセージの全文
- 実行したコマンド
- 環境情報（OS、バージョン等）

---

## 📅 更新履歴

| 日付 | バージョン | 内容 |
|------|-----------|------|
| 2025-11-02 | 1.0 | 初版リリース |

---

## 🎉 次のステップ

すべてのドキュメントを完了したら、以下を検討してください：

1. **パフォーマンス最適化**
   - 推論時間の測定
   - メモリ使用量の削減
   - バッチ推論の実装

2. **AUTOSAR環境への展開**
   - クロスコンパイル環境の構築
   - リアルタイム性の検証
   - 安全性の考慮

3. **他のモデルへの応用**
   - 分類問題
   - 異常検知
   - 画像処理

---

**Good Luck！** 🚀

---

**作成日**: 2025-11-02  
**バージョン**: 1.0  
**メンテナー**: ML Study Project Team

