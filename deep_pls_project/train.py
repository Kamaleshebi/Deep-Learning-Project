"""
train.py — CLI training script for one dataset at a time.

Usage
-----
    python train.py --dataset butane   --layers 3 --components 3
    python train.py --dataset concrete --layers 3 --components 4
"""

import os, sys, argparse, pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data_loader    import load_dataset
from residual_model import DeepResidualPLS, BaselinePLS
from evaluate       import evaluate_model, compare_models


def get_args():
    p = argparse.ArgumentParser(description="Train Deep Residual PLS")
    p.add_argument("--dataset",     type=str,   default="butane",
                   choices=["butane", "concrete"],
                   help="'butane' = Butane_data.csv | 'concrete' = Concrete_Data.csv")
    p.add_argument("--layers",      type=int,   default=3,
                   help="Hidden PLS layers L (default: 3)")
    p.add_argument("--components",  type=int,   default=3,
                   help="Supervised latent variables qs per layer (default: 3)")
    p.add_argument("--final_comp",  type=int,   default=2,
                   help="Fusion layer components qs(d) (default: 2)")
    p.add_argument("--test_size",   type=float, default=0.5)
    p.add_argument("--data_dir",    type=str,   default="data")
    p.add_argument("--results_dir", type=str,   default="results")
    return p.parse_args()


def train(args):
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  DEEP RESIDUAL PLS | Dataset: {args.dataset.upper()}")
    print(f"{'='*60}")

    X_train, X_test, Y_train, Y_test, sx, sy, feature_cols = load_dataset(
        args.dataset, data_dir=args.data_dir, test_size=args.test_size)

    print("\n[1/2] Training Baseline PLS ...")
    baseline = BaselinePLS(n_components=args.components)
    baseline.fit(X_train, Y_train)
    Y_pred_base = baseline.predict(X_test)

    print(f"\n[2/2] Training Deep Residual PLS (L={args.layers}) ...")
    deep = DeepResidualPLS(
        n_layers=args.layers,
        n_components_layer=args.components,
        n_components_final=args.final_comp,
    )
    deep.fit(X_train, Y_train)
    Y_pred_deep = deep.predict(X_test)

    actual_layers = len(deep._layers)
    layer_preds = {l: deep.predict_layer(X_test, l) for l in range(1, actual_layers+1)}
    res_base = evaluate_model("Baseline PLS",      Y_test, Y_pred_base)
    res_deep = evaluate_model("Deep Residual PLS", Y_test, Y_pred_deep)
    layer_res = [evaluate_model(f"Layer {l}", Y_test, layer_preds[l])
                 for l in layer_preds]
    compare_models([res_base] + layer_res + [res_deep])

    pct = (res_base["rmse"] - res_deep["rmse"]) / res_base["rmse"] * 100
    print(f"\n  RMSE improvement: {pct:+.1f}%")

    prefix = os.path.join(args.results_dir, args.dataset)
    np.save(f"{prefix}_Y_test.npy",      Y_test)
    np.save(f"{prefix}_Y_pred_base.npy", Y_pred_base)
    np.save(f"{prefix}_Y_pred_deep.npy", Y_pred_deep)
    with open(f"{prefix}_deep_model.pkl",    "wb") as f: pickle.dump(deep,     f)
    with open(f"{prefix}_baseline_model.pkl","wb") as f: pickle.dump(baseline, f)
    print(f"\n  Saved to '{args.results_dir}/'")


if __name__ == "__main__":
    train(get_args())
