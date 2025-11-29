#!/bin/bash

# ONNX C++推論の実行スクリプト

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ビルド済みかチェック
if [ ! -f "${SCRIPT_DIR}/build/onnx_inference" ]; then
    echo "❌ 実行ファイルが見つかりません"
    echo "先にビルドを実行してください:"
    echo "  ./setup_and_build.sh"
    exit 1
fi

# cpp_inferenceディレクトリに移動
cd "${SCRIPT_DIR}"

# 推論実行
echo "========================================"
echo "  ONNX推論実行"
echo "========================================"
echo ""

# ライブラリパスを設定して実行
LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH ./build/onnx_inference

echo ""
echo "✓ 実行完了"

