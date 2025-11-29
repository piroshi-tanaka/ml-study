"""Evaluate ONNX conversion accuracy for a HistGradientBoostingRegressor.

This script trains a time-series style regressor, converts it to ONNX,
and compares prediction accuracy between the original scikit-learn model
and the ONNX Runtime inference results. It is designed to be runnable with
either a synthetic dataset (default) or a CSV time-series dataset such as
those commonly found on Kaggle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def generate_synthetic_series(n_samples: int = 720, noise: float = 0.05, seed: int = 0) -> np.ndarray:
    """Create a simple seasonal time series with trend and noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float64)
    seasonal = np.sin(2 * np.pi * t / 24) + 0.5 * np.sin(2 * np.pi * t / 168)
    trend = 0.0015 * t
    return 5 + trend + seasonal + rng.normal(scale=noise, size=n_samples)


def series_to_supervised(series: Iterable[float], lags: int) -> Tuple[np.ndarray, np.ndarray]:
    """Turn a univariate series into a supervised learning matrix."""
    arr = np.asarray(series, dtype=np.float64).ravel()
    if arr.shape[0] <= lags:
        raise ValueError(f"Need more than {lags} samples to build lag features, got {arr.shape[0]}.")
    X, y = [], []
    for i in range(lags, len(arr)):
        X.append(arr[i - lags : i])
        y.append(arr[i])
    return np.vstack(X), np.asarray(y)


def load_series_from_csv(csv_path: Path, value_col: str, time_col: str | None = None) -> np.ndarray:
    """Load a time series from CSV, optionally sorting by a time column."""
    df = pd.read_csv(csv_path)
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in {csv_path}. Available: {list(df.columns)}")
    if time_col and time_col in df.columns:
        df = df.sort_values(time_col)
    return df[value_col].to_numpy(dtype=np.float64)


def split_train_test(X: np.ndarray, y: np.ndarray, test_size: float) -> Tuple[np.ndarray, ...]:
    """Time-ordered split (no shuffling)."""
    n_test = max(1, int(len(X) * test_size))
    split = len(X) - n_test
    return X[:split], X[split:], y[:split], y[split:]


def train_model(X_train: np.ndarray, y_train: np.ndarray, random_state: int) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=400,
        l2_regularization=0.0,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def convert_model_to_onnx(model: HistGradientBoostingRegressor, n_features: int) -> onnx.ModelProto:
    """Convert scikit-learn model to ONNX with float32 inputs."""
    initial_types = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(model, initial_types=initial_types, target_opset=17)
    return onnx_model


def run_onnx_inference(onnx_model: onnx.ModelProto, X: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    preds = sess.run(None, {input_name: X.astype(np.float32, copy=False)})[0]
    return preds.ravel()


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def compare_models(
    model: HistGradientBoostingRegressor, onnx_model: onnx.ModelProto, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, Dict[str, float]]:
    preds_sklearn = model.predict(X_test)
    preds_sklearn_fp32 = model.predict(X_test.astype(np.float32))
    preds_onnx = run_onnx_inference(onnx_model, X_test)

    metrics = {
        "sklearn_float64": evaluate_predictions(y_test, preds_sklearn),
        "sklearn_float32": evaluate_predictions(y_test, preds_sklearn_fp32),
        "onnx_runtime": evaluate_predictions(y_test, preds_onnx),
    }

    diff_vs_sklearn = {
        "mean_abs_diff_to_sklearn64": float(np.mean(np.abs(preds_sklearn - preds_onnx))),
        "max_abs_diff_to_sklearn64": float(np.max(np.abs(preds_sklearn - preds_onnx))),
        "mean_abs_diff_to_sklearn32": float(np.mean(np.abs(preds_sklearn_fp32 - preds_onnx))),
        "max_abs_diff_to_sklearn32": float(np.max(np.abs(preds_sklearn_fp32 - preds_onnx))),
    }
    metrics["differences"] = diff_vs_sklearn
    return metrics


def save_outputs(output_dir: Path, onnx_model: onnx.ModelProto, metrics: Dict[str, Dict[str, float]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "hist_gradient_boosting.onnx"
    metrics_path = output_dir / "metrics.json"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved ONNX model to {onnx_path}")
    print(f"Saved metrics to {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare scikit-learn and ONNXRuntime predictions.")
    parser.add_argument("--csv-path", type=Path, help="Path to a CSV file containing a time series.")
    parser.add_argument(
        "--value-col",
        type=str,
        default="value",
        help="Column name of the numeric series to forecast when using a CSV dataset.",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default=None,
        help="Optional column to sort by time when reading CSV data.",
    )
    parser.add_argument("--lags", type=int, default=48, help="Number of lag features to create for the series.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size ratio (time-ordered split).")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory to save ONNX and metrics.")
    parser.add_argument("--random-state", type=int, default=0, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.csv_path:
        series = load_series_from_csv(args.csv_path, value_col=args.value_col, time_col=args.time_col)
        dataset_desc = f"CSV:{args.csv_path}"
    else:
        series = generate_synthetic_series(seed=args.random_state)
        dataset_desc = "synthetic seasonal series"

    X, y = series_to_supervised(series, lags=args.lags)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=args.test_size)

    model = train_model(X_train, y_train, random_state=args.random_state)
    onnx_model = convert_model_to_onnx(model, n_features=X_train.shape[1])
    metrics = compare_models(model, onnx_model, X_test, y_test)

    print(f"Dataset: {dataset_desc}")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    save_outputs(args.output_dir, onnx_model, metrics)


if __name__ == "__main__":
    main()
