"""
run_all.py  ←  MAIN FILE TO RUN
=================================
Trains and evaluates the Deep Residual PLS on both datasets,
then generates all plots.

Usage
-----
    python run_all.py                      # both datasets
    python run_all.py --dataset butane     # Butane only
    python run_all.py --dataset concrete   # Concrete only

Input files required (place in data/ folder)
---------------------------------------------
    data/Butane_data.csv        ← Debutanizer column (2000 rows, 7 inputs)
    data/Concrete_Data.csv      ← Cement clinker     (1030 rows, 8 inputs)
"""

import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from data_loader    import load_dataset
from residual_model import DeepResidualPLS, BaselinePLS
from evaluate       import evaluate_model, compare_models, rmse
from visualize      import (plot_actual_vs_predicted, plot_parity,
                             plot_error_distribution, plot_layerwise_rmse,
                             plot_with_uncertainty, plot_dashboard)

# ── Dataset configs (matching paper settings) ─────────────────────────────────
CONFIGS = {
    "butane": dict(
        csv_name   = "Butane_data.csv",
        layers     = 3,
        components = 3,
        final_comp = 2,
        test_size  = 0.5,
        display    = "Debutanizer Column - Butane Content",
    ),
    "concrete": dict(
        csv_name   = "Concrete_Data.csv",
        layers     = 3,
        components = 4,
        final_comp = 3,
        test_size  = 0.5,
        display    = "Concrete Compressive Strength",
    ),
}


def run_dataset(name: str, cfg: dict):
    print("\n" + "="*65)
    print(f"  DATASET : {cfg['display']}")
    print(f"  FILE    : data/{cfg['csv_name']}")
    print(f"  MODEL   : {cfg['layers']} layers | "
          f"{cfg['components']} components/layer | "
          f"{cfg['final_comp']} fusion components")
    print("="*65)

    # 1. Load data
    X_train, X_test, Y_train, Y_test, sx, sy, feature_cols = load_dataset(
        name, data_dir="data", test_size=cfg["test_size"])

    # 2. Baseline single-layer PLS
    print("\n[Step 1/2]  Training Baseline (single-layer) PLS ...")
    baseline = BaselinePLS(n_components=cfg["components"])
    baseline.fit(X_train, Y_train)
    Y_pred_base = baseline.predict(X_test)

    # 3. Deep Residual PLS
    print(f"\n[Step 2/2]  Training Deep Residual PLS "
          f"(L={cfg['layers']}, qs={cfg['components']}, fusion={cfg['final_comp']}) ...")
    deep = DeepResidualPLS(
        n_layers           = cfg["layers"],
        n_components_layer = cfg["components"],
        n_components_final = cfg["final_comp"],
    )
    deep.fit(X_train, Y_train)
    Y_pred_deep = deep.predict(X_test)

    # 4. Layer-by-layer predictions
    actual_layers = len(deep._layers)
    layer_preds   = {l: deep.predict_layer(X_test, l) for l in range(1, actual_layers+1)}

    # 5. Evaluate
    print("\n" + "-"*65)
    print("  TEST-SET RESULTS")
    print("-"*65)
    res_base = evaluate_model("Baseline PLS",      Y_test, Y_pred_base)
    res_deep = evaluate_model("Deep Residual PLS", Y_test, Y_pred_deep)
    layer_results = [
        evaluate_model(f"Layer {l} (standalone)", Y_test, layer_preds[l])
        for l in layer_preds
    ]
    compare_models([res_base] + layer_results + [res_deep])

    pct_rmse = (res_base["rmse"] - res_deep["rmse"]) / res_base["rmse"] * 100
    pct_r2   = (res_deep["r2"]  - res_base["r2"])  / (abs(res_base["r2"]) + 1e-8) * 100
    print(f"\n  >> RMSE improvement : {pct_rmse:+.1f}%  (paper reports 10-20%)")
    print(f"  >> R²   improvement : {pct_r2:+.1f}%")

    # 6. Save results
    os.makedirs("results", exist_ok=True)
    prefix = os.path.join("results", name)
    np.save(f"{prefix}_Y_test.npy",      Y_test)
    np.save(f"{prefix}_Y_pred_base.npy", Y_pred_base)
    np.save(f"{prefix}_Y_pred_deep.npy", Y_pred_deep)
    for l in layer_preds:
        np.save(f"{prefix}_Y_pred_layer{l}.npy", layer_preds[l])
    with open(f"{prefix}_summary.txt", "w") as f:
        f.write(
            f"Dataset : {cfg['display']}\n"
            f"CSV     : data/{cfg['csv_name']}\n"
            f"Layers  : {actual_layers}\n"
            f"Components/layer : {cfg['components']}\n"
            f"Fusion components: {cfg['final_comp']}\n\n"
            f"Baseline  RMSE={res_base['rmse']:.5f}  R2={res_base['r2']:.5f}\n"
            f"Deep PLS  RMSE={res_deep['rmse']:.5f}  R2={res_deep['r2']:.5f}\n"
            f"RMSE improvement: {pct_rmse:+.1f}%\n"
        )

    # 7. Generate plots
    print("\n  Generating plots ...")
    os.makedirs("outputs", exist_ok=True)
    dn = cfg["display"]
    layer_rmse_dict = {l: rmse(Y_test, layer_preds[l]) for l in layer_preds}
    plot_actual_vs_predicted(Y_test, Y_pred_base, Y_pred_deep, dn, n_show=200)
    plot_parity             (Y_test, Y_pred_base, Y_pred_deep, dn)
    plot_error_distribution (Y_test, Y_pred_base, Y_pred_deep, dn)
    plot_layerwise_rmse     (layer_rmse_dict, res_base["rmse"], res_deep["rmse"], dn)
    plot_with_uncertainty   (Y_test, Y_pred_base, Y_pred_deep, dn, n_show=100)
    plot_dashboard          (Y_test, Y_pred_base, Y_pred_deep, layer_preds,
                             res_base, res_deep, dn)
    print(f"  All plots saved to outputs/")


def main():
    p = argparse.ArgumentParser(
        description="Deep Residual PLS — Main runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py                      # run both datasets
  python run_all.py --dataset butane     # Butane only
  python run_all.py --dataset concrete   # Concrete only

Required CSV files:
  data/Butane_data.csv       (2000 rows, 7 inputs → Butane content)
  data/Concrete_Data.csv     (1030 rows, 8 inputs → Compressive strength)
        """
    )
    p.add_argument("--dataset", default="both",
                   choices=["butane", "concrete", "both"])
    args = p.parse_args()

    datasets = ["butane", "concrete"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        run_dataset(ds, CONFIGS[ds])

    print("\n" + "="*65)
    print("  DONE — outputs/ for plots | results/ for saved data")
    print("="*65)


if __name__ == "__main__":
    main()
