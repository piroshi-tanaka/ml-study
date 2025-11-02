# クイックスタートガイド

## 最短手順（WSL Ubuntu）

### 前提条件

- Windows 10/11でWSLが有効化されている
- WSL Ubuntu（18.04以降）がインストールされている
- Python側でJupyter Notebookを実行済み（テストデータが生成されている）

### ステップ1: WSLを起動

PowerShellまたはコマンドプロンプトで：

```bash
wsl
```

### ステップ2: プロジェクトディレクトリへ移動

```bash
cd /mnt/c/workspace/src/ml-study/scikit_learn_onnx_repo/cpp_inference
```

### ステップ3: セットアップスクリプトに実行権限を付与

```bash
chmod +x setup_and_build.sh run_inference.sh
```

### ステップ4: 自動セットアップとビルド

```bash
./setup_and_build.sh
```

このスクリプトは以下を自動実行します：
- 必要なパッケージのインストール（build-essential, cmake, wget）
- ONNX Runtime C++ライブラリのダウンロードとインストール
- プロジェクトのビルド

**注意**: 初回実行時は管理者権限（sudo）のパスワードを求められます。

### ステップ5: 推論実行

```bash
./run_inference.sh
```

または直接実行：

```bash
cd build
./onnx_inference ../time_series_model.onnx ../test_data.csv ../test_labels.csv
```

## 期待される結果

✅ **成功時の出力例**:

```
========================================
  ONNX時系列予測推論（C++版）
========================================

📂 ファイル読み込み中...
...
✓ 推論完了

【精度評価】
【C++ ONNX推論】
  RMSE: XX.XXXXXX
  MAE:  XX.XXXXXX
  R²:   X.XXXXXX

【Python ONNX vs C++ ONNX】
  最大差分: X.XXXXXXe-XX
  平均差分: X.XXXXXXe-XX
  ✓ Python ONNXとC++ ONNXの予測はほぼ一致！
```

## トラブルシューティング

### Q1: `setup_and_build.sh`の実行中にエラーが発生

**A1**: エラーメッセージを確認してください。よくある原因：
- インターネット接続が不安定 → 再実行
- ディスク容量不足 → 空き容量を確保
- パッケージのインストールに失敗 → `sudo apt update`を再実行

### Q2: 実行時に「libonnxruntime.so が見つからない」エラー

**A2**: ライブラリパスが通っていません：
```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
./run_inference.sh
```

### Q3: ファイルが見つからないエラー

**A3**: Python側でJupyter Notebookを実行し、テストデータを生成してください：
- `time_series_model.onnx`
- `test_data.csv`
- `test_labels.csv`

これらのファイルが`cpp_inference/`ディレクトリに存在することを確認：
```bash
ls -la
```

### Q4: WSLのパスがわからない

**A4**: Windowsのパスから変換：
- `C:\workspace\...` → `/mnt/c/workspace/...`
- `D:\project\...` → `/mnt/d/project/...`

## 手動セットアップ（スクリプトを使わない場合）

詳細は`README.md`を参照してください。

## 次のステップ

1. **精度検証**: Python推論との差分を確認
2. **パフォーマンス測定**: 推論時間を計測
3. **最適化**: モデルサイズや推論速度の改善
4. **AUTOSAR展開**: ターゲット環境への移植

## サポート

問題が解決しない場合は、`README.md`の詳細なトラブルシューティングセクションを参照してください。

