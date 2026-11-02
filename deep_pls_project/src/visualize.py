"""
visualize.py
============
All plotting functions for the Deep Residual PLS project.
Saves figures to outputs/ and optionally displays them.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

def _safe_name(name):
    """Make a Windows + Linux safe filename — strips all illegal chars."""
    import re as _re
    safe = name.lower()
    # Characters illegal on Windows: \ / : * ? " < > | ( ) and whitespace
    safe = _re.sub('[' + r'\/:*?"<>|()\s' + ']+', '_', safe)
    safe = _re.sub('_+', '_', safe).strip('_')
    return safe


PALETTE = {
    "actual":   "#2c3e50",
    "baseline": "#e74c3c",
    "deep":     "#2980b9",
    "layer1":   "#8e44ad",
    "layer2":   "#27ae60",
    "layer3":   "#f39c12",
    "layer4":   "#16a085",
    "layer5":   "#d35400",
}


# ── 1. Actual vs Predicted ────────────────────────────────────────────────────

def plot_actual_vs_predicted(Y_test, Y_pred_base, Y_pred_deep,
                              dataset_name="Dataset", save_dir="outputs",
                              n_show=None):
    """
    Two-panel time-series plot: baseline vs deep.
    Mirrors Fig. 3 in the paper.
    """
    os.makedirs(save_dir, exist_ok=True)
    Y_test  = Y_test.ravel()
    Y_base  = Y_pred_base.ravel()
    Y_deep  = Y_pred_deep.ravel()
    n = len(Y_test) if n_show is None else min(n_show, len(Y_test))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Actual vs Predicted — {dataset_name}", fontsize=14, fontweight="bold")

    for ax, pred, label, color in [
        (axes[0], Y_base, "Baseline PLS",      PALETTE["baseline"]),
        (axes[1], Y_deep, "Deep Residual PLS", PALETTE["deep"]),
    ]:
        ax.plot(Y_test[:n], color=PALETTE["actual"], lw=1.2, label="Actual", zorder=3)
        ax.plot(pred[:n],   color=color,             lw=1.0, alpha=0.85,
                linestyle="--", label=label, zorder=2)
        ax.set_ylabel("Output value", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"({label})", fontsize=10)

    axes[1].set_xlabel("Sample index", fontsize=10)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{_safe_name(dataset_name)}_actual_vs_pred.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── 2. Parity plot ────────────────────────────────────────────────────────────

def plot_parity(Y_test, Y_pred_base, Y_pred_deep,
                dataset_name="Dataset", save_dir="outputs"):
    """
    Scatter parity plot (predicted vs actual). Mirrors Fig. 4 in the paper.
    """
    os.makedirs(save_dir, exist_ok=True)
    Y_test = Y_test.ravel()
    vmin, vmax = Y_test.min(), Y_test.max()
    margin = (vmax - vmin) * 0.05

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Parity Plot — {dataset_name}", fontsize=13, fontweight="bold")

    for ax, pred, title, color in [
        (axes[0], Y_pred_base.ravel(), "Baseline PLS",      PALETTE["baseline"]),
        (axes[1], Y_pred_deep.ravel(), "Deep Residual PLS", PALETTE["deep"]),
    ]:
        ax.scatter(pred, Y_test, color=color, alpha=0.4, s=18, edgecolors="none")
        diag = np.linspace(vmin - margin, vmax + margin, 100)
        ax.plot(diag, diag, "k--", lw=1.2, label="Perfect fit")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual",    fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(vmin - margin, vmax + margin)
        ax.set_ylim(vmin - margin, vmax + margin)
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    path = os.path.join(save_dir, f"{_safe_name(dataset_name)}_parity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── 3. Error distribution ─────────────────────────────────────────────────────

def plot_error_distribution(Y_test, Y_pred_base, Y_pred_deep,
                             dataset_name="Dataset", save_dir="outputs"):
    """Histogram of prediction residuals for both models."""
    os.makedirs(save_dir, exist_ok=True)
    err_base = (Y_pred_base - Y_test).ravel()
    err_deep = (Y_pred_deep - Y_test).ravel()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    fig.suptitle(f"Prediction Error Distribution — {dataset_name}",
                 fontsize=13, fontweight="bold")

    for ax, err, title, color in [
        (axes[0], err_base, "Baseline PLS",      PALETTE["baseline"]),
        (axes[1], err_deep, "Deep Residual PLS", PALETTE["deep"]),
    ]:
        ax.hist(err, bins=50, color=color, alpha=0.75, edgecolor="white", lw=0.3)
        ax.axvline(0,   color="black", lw=1.5, linestyle="--")
        ax.axvline(err.mean(), color="orange", lw=1.2, linestyle=":", label=f"mean={err.mean():.4f}")
        ax.set_xlabel("Prediction error", fontsize=10)
        ax.set_ylabel("Count",            fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{_safe_name(dataset_name)}_error_dist.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── 4. Layer-wise RMSE comparison ─────────────────────────────────────────────

def plot_layerwise_rmse(layer_rmse_dict, baseline_rmse, deep_rmse,
                        dataset_name="Dataset", save_dir="outputs"):
    """
    Bar chart of RMSE per layer + baseline + deep fusion.
    Mirrors Table II & VI in the paper.

    Parameters
    ----------
    layer_rmse_dict : dict  e.g. {1: 0.176, 2: 0.156, 3: 0.171}
    """
    os.makedirs(save_dir, exist_ok=True)
    layers  = sorted(layer_rmse_dict.keys())
    rmses   = [layer_rmse_dict[l] for l in layers]
    labels  = [f"Layer {l}" for l in layers] + ["Baseline PLS", "Deep PPLS"]
    values  = rmses + [baseline_rmse, deep_rmse]
    colors  = [list(PALETTE.values())[i % len(PALETTE)] for i in range(len(layers))] \
              + [PALETTE["baseline"], PALETTE["deep"]]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.3), 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", lw=0.8, width=0.6)
    ax.set_ylabel("RMSE", fontsize=11)
    ax.set_title(f"Layer-wise RMSE Comparison — {dataset_name}",
                 fontsize=12, fontweight="bold")
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.grid(True, axis="y", alpha=0.35, linestyle="--")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Highlight Deep PPLS bar
    bars[-1].set_edgecolor("black")
    bars[-1].set_linewidth(2)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{_safe_name(dataset_name)}_layerwise_rmse.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── 5. Prediction with uncertainty bands ─────────────────────────────────────

def plot_with_uncertainty(Y_test, Y_pred_base, Y_pred_deep,
                          dataset_name="Dataset", save_dir="outputs",
                          n_show=100):
    """
    Show prediction with ±1 std uncertainty bands.
    Mirrors Fig. 5 & 9 in the paper.
    We compute rolling std of residuals as a proxy for local uncertainty.
    """
    os.makedirs(save_dir, exist_ok=True)
    n = min(n_show, len(Y_test))
    yt = Y_test.ravel()[:n]

    def rolling_std(arr, w=10):
        out = np.zeros_like(arr)
        for i in range(len(arr)):
            start = max(0, i - w)
            out[i] = arr[start:i+1].std()
        return out

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Predictions with Uncertainty (first {n} samples) — {dataset_name}",
                 fontsize=13, fontweight="bold")

    for ax, pred_full, title, color in [
        (axes[0], Y_pred_base, "Baseline PLS",      PALETTE["baseline"]),
        (axes[1], Y_pred_deep, "Deep Residual PLS", PALETTE["deep"]),
    ]:
        pred = pred_full.ravel()[:n]
        err  = np.abs(pred - yt)
        std  = rolling_std(err, w=10)
        ax.plot(yt,   color=PALETTE["actual"], lw=1.5, label="Actual", zorder=3)
        ax.plot(pred, color=color, lw=1.0, linestyle="--", label=title, zorder=2)
        ax.fill_between(np.arange(n), pred - std, pred + std,
                        color=color, alpha=0.18, label="±1σ uncertainty")
        ax.legend(fontsize=9)
        ax.set_ylabel("Output value", fontsize=10)
        ax.set_title(f"({title})", fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel("Sample index", fontsize=10)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{_safe_name(dataset_name)}_uncertainty.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── 6. Comprehensive summary dashboard ────────────────────────────────────────

def plot_dashboard(Y_test, Y_pred_base, Y_pred_deep, layer_preds,
                   res_base, res_deep, dataset_name="Dataset", save_dir="outputs"):
    """
    Single-figure dashboard with 6 panels.
    """
    os.makedirs(save_dir, exist_ok=True)
    yt    = Y_test.ravel()
    yb    = Y_pred_base.ravel()
    yd    = Y_pred_deep.ravel()
    n100  = min(100, len(yt))

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"Deep Residual PLS — {dataset_name} Dashboard",
                 fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panel A: time-series baseline
    ax_a = fig.add_subplot(gs[0, :2])
    ax_a.plot(yt[:n100],  color=PALETTE["actual"],   lw=1.5, label="Actual")
    ax_a.plot(yb[:n100],  color=PALETTE["baseline"], lw=1.0, ls="--", alpha=0.85, label="Baseline PLS")
    ax_a.set_title("(A) Baseline PLS Prediction", fontsize=11)
    ax_a.legend(fontsize=9); ax_a.grid(True, alpha=0.3)
    ax_a.set_xlabel("Sample"); ax_a.set_ylabel("Output")

    # Panel B: time-series deep
    ax_b = fig.add_subplot(gs[1, :2])
    ax_b.plot(yt[:n100],  color=PALETTE["actual"], lw=1.5, label="Actual")
    ax_b.plot(yd[:n100],  color=PALETTE["deep"],   lw=1.0, ls="--", alpha=0.85, label="Deep Residual PLS")
    ax_b.set_title("(B) Deep Residual PLS Prediction", fontsize=11)
    ax_b.legend(fontsize=9); ax_b.grid(True, alpha=0.3)
    ax_b.set_xlabel("Sample"); ax_b.set_ylabel("Output")

    # Panel C: parity
    ax_c = fig.add_subplot(gs[2, :2])
    vmin, vmax = yt.min(), yt.max()
    mg = (vmax - vmin) * 0.05
    ax_c.scatter(yb, yt, color=PALETTE["baseline"], alpha=0.35, s=12, label="Baseline")
    ax_c.scatter(yd, yt, color=PALETTE["deep"],     alpha=0.35, s=12, label="Deep PLS")
    diag = np.linspace(vmin-mg, vmax+mg, 100)
    ax_c.plot(diag, diag, "k--", lw=1.2)
    ax_c.set_title("(C) Parity Plot", fontsize=11)
    ax_c.set_xlabel("Predicted"); ax_c.set_ylabel("Actual")
    ax_c.legend(fontsize=9); ax_c.grid(True, alpha=0.3)

    # Panel D: error histograms
    ax_d = fig.add_subplot(gs[0, 2])
    ax_d.hist((yb-yt), bins=40, color=PALETTE["baseline"], alpha=0.6, label="Baseline")
    ax_d.hist((yd-yt), bins=40, color=PALETTE["deep"],     alpha=0.6, label="Deep")
    ax_d.axvline(0, color="black", lw=1.2, ls="--")
    ax_d.set_title("(D) Error Distribution", fontsize=11)
    ax_d.set_xlabel("Error"); ax_d.legend(fontsize=8); ax_d.grid(True, alpha=0.3)

    # Panel E: layer-wise RMSE bar
    ax_e = fig.add_subplot(gs[1, 2])
    from evaluate import rmse as _rmse
    layer_keys = sorted(layer_preds.keys())
    lbls   = [f"L{l}" for l in layer_keys] + ["Base", "Deep"]
    vals   = [_rmse(Y_test, layer_preds[l]) for l in layer_keys] \
             + [res_base["rmse"], res_deep["rmse"]]
    clrs   = [PALETTE.get(f"layer{l}", "#aaa") for l in layer_keys] \
             + [PALETTE["baseline"], PALETTE["deep"]]
    bars = ax_e.bar(lbls, vals, color=clrs, edgecolor="white")
    bars[-1].set_edgecolor("black"); bars[-1].set_linewidth(2)
    ax_e.set_title("(E) Layer-wise RMSE", fontsize=11)
    ax_e.set_ylabel("RMSE"); ax_e.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax_e.text(bar.get_x()+bar.get_width()/2, v+max(vals)*0.01,
                  f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    # Panel F: metrics summary table
    ax_f = fig.add_subplot(gs[2, 2])
    ax_f.axis("off")
    col_labels = ["Model", "RMSE", "R²", "MAE"]
    from evaluate import mae as _mae
    table_data = [
        ["Baseline", f"{res_base['rmse']:.4f}", f"{res_base['r2']:.4f}", f"{res_base['mae']:.4f}"],
        ["Deep PLS", f"{res_deep['rmse']:.4f}", f"{res_deep['r2']:.4f}", f"{res_deep['mae']:.4f}"],
    ]
    tbl = ax_f.table(cellText=table_data, colLabels=col_labels,
                     cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 2.0)
    ax_f.set_title("(F) Metrics Summary", fontsize=11, pad=15)

    path = os.path.join(save_dir, f"{_safe_name(dataset_name)}_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
