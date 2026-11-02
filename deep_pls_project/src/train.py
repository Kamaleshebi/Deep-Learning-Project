"""
train.py
========
Main training script for the Deep Residual Probabilistic PLS soft sensor.

Usage (from project root):
    python src/train.py --dataset butane --layers 3 --components 3
    python src/train.py --dataset concrete --layers 5 --components 10
"""

import sys
import os
import argparse
import json
import pickle
import numpy as np

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_dataset
from residual_model import DeepResidualPLS, BaselinePLS
from evaluate import evaluate_model, compare_models


def parse_args():
    p = argparse.ArgumentParser(description="Train Deep Residual PLS Soft Sensor")
    p.add_argument("--dataset",    type=str, default="concrete",
                   choices=["butane", "concrete"],
                   help="Which dataset to train on")
    p.add_argument("--layers",     type=int, default=3,
                   help="Number of hidden PLS layers (L in paper)")
    p.add_argument("--components", type=int, default=3,
                   help="Latent variables per hidden layer (qs)")
    p.add_argument("--components_final", type=int, default=2,
                   help="Latent variables in fusion layer")
    p.add_argument("--test_size",  type=float, default=0.5,
                   help="Fraction of data used for testing")
    p.add_argument("--data_dir",   type=str, default="data",
                   help="Path to data directory")
    p.add_argument("--output_dir", type=str, default="results",
                   help="Directory for saving results")
    p.add_argument("--save_model", action="store_true",
                   help="Pickle the trained model")
    return p.parse_args()


def train(dataset: str,
          n_layers: int = 3,
          n_components: int = 3,
          n_components_final: int = 2,
          test_size: float = 0.5,
          data_dir: str = "data",
          output_dir: str = "results",
          save_model: bool = False):
    """
    Full training + evaluation pipeline.

    Returns
    -------
    dict with predictions and metrics for downstream use (e.g. notebook).
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DEEP RESIDUAL PLS  |  dataset={dataset}")
    print(f"  layers={n_layers}  |  components={n_components}")
    print(f"{'='*60}")

    X_train, X_test, Y_train, Y_test, scaler_X, scaler_Y, feature_cols = load_dataset(
        dataset, data_dir=data_dir, test_size=test_size
    )

    # ── 2. Baseline single-layer PLS ─────────────────────────────────────────
    print("\n[Baseline] Training single-layer PLS …")
    baseline = BaselinePLS(n_components=n_components)
    baseline.fit(X_train, Y_train)
    Y_pred_baseline = baseline.predict(X_test)

    # ── 3. Deep Residual PLS ─────────────────────────────────────────────────
    print(f"\n[Deep PLS] Training {n_layers}-layer Deep Residual PLS …")
    deep_model = DeepResidualPLS(
        n_layers=n_layers,
        n_components_layer=n_components,
        n_components_final=n_components_final,
    )
    deep_model.fit(X_train, Y_train)
    Y_pred_deep = deep_model.predict(X_test)

    # ── 4. Layer-wise predictions (for diagnostics) ──────────────────────────
    layer_preds = {}
    for l in range(1, n_layers + 1):
        layer_preds[f"layer_{l}"] = deep_model.predict_layer(X_test, layer_id=l)

    # ── 5. Evaluation ─────────────────────────────────────────────────────────
    print("\n[Evaluation] Test set metrics:")
    r_baseline = evaluate_model("Baseline PLS (shallow)", Y_test, Y_pred_baseline)
    r_deep     = evaluate_model(f"Deep Residual PLS (L={n_layers})", Y_test, Y_pred_deep)

    layer_results = []
    for l in range(1, n_layers + 1):
        r_l = evaluate_model(f"Layer {l} (standalone)", Y_test, layer_preds[f"layer_{l}"])
        layer_results.append(r_l)

    all_results = [r_baseline] + layer_results + [r_deep]
    compare_models(all_results)

    # ── 6. RMSE improvement ───────────────────────────────────────────────────
    improvement = (r_baseline["rmse"] - r_deep["rmse"]) / r_baseline["rmse"] * 100
    print(f"\n  ✅ RMSE improvement of Deep PLS over Baseline: {improvement:.1f}%")
    print(f"  ✅ R² improvement: {r_deep['r2'] - r_baseline['r2']:.4f}")

    # ── 7. Save results ───────────────────────────────────────────────────────
    results_path = os.path.join(output_dir, f"{dataset}_results.json")
    save_data = {
        "dataset": dataset,
        "config": {
            "n_layers": n_layers,
            "n_components": n_components,
            "n_components_final": n_components_final,
        },
        "metrics": {
            "baseline": r_baseline,
            "deep_pls": r_deep,
            "layers": layer_results,
        },
        "rmse_improvement_pct": round(improvement, 2),
    }
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved → {results_path}")

    if save_model:
        model_path = os.path.join(output_dir, f"{dataset}_deep_pls.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({"baseline": baseline, "deep_model": deep_model}, f)
        print(f"  Model saved   → {model_path}")

    # ── 8. Save predictions (for plotting) ────────────────────────────────────
    pred_path = os.path.join(output_dir, f"{dataset}_predictions.npz")
    np.savez(
        pred_path,
        Y_test=Y_test,
        Y_pred_baseline=Y_pred_baseline,
        Y_pred_deep=Y_pred_deep,
        **{f"Y_pred_layer_{l}": layer_preds[f"layer_{l}"] for l in range(1, n_layers + 1)},
    )
    print(f"  Predictions   → {pred_path}")

    return {
        "Y_test": Y_test,
        "Y_pred_baseline": Y_pred_baseline,
        "Y_pred_deep": Y_pred_deep,
        "layer_preds": layer_preds,
        "metrics": save_data["metrics"],
        "models": {"baseline": baseline, "deep_model": deep_model},
        "improvement_pct": improvement,
    }


if __name__ == "__main__":
    args = parse_args()
    train(
        dataset=args.dataset,
        n_layers=args.layers,
        n_components=args.components,
        n_components_final=args.components_final,
        test_size=args.test_size,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        save_model=args.save_model,
    )
