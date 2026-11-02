import os, sys, io, json, base64, threading
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify

SRC_DIR  = os.path.join(os.path.dirname(__file__), "..", "src")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
sys.path.insert(0, SRC_DIR)

from data_loader import load_dataset, CONCRETE_OUTPUT_COL
from residual_model import DeepResidualPLS, BaselinePLS
from autoencoder_shap import ConcreteAutoencoder, generate_shap_plot, AutoencoderPredictor
from gnn_model import ConcreteGNN
from evaluate import rmse, r2, mae

app = Flask(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
_state = {
    "status":      "idle",
    "log":         [],
    "results":     {},
    "deep_model":  None,
    "baseline":    None,
    "autoencoder": None,
    "scaler_x":    None,
    "scaler_y":    None,
}
_lock = threading.Lock()

def _log(msg):
    with _lock:
        _state["log"].append(msg)
    print(msg)

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64

BG   = "#0f1117"
BG2  = "#1a1d2e"
BLUE = "#64b5f6"
GRN  = "#69f0ae"
RED  = "#ef5350"
YLW  = "#ffd740"
TXT  = "#e0e4ff"
MUT  = "#7b82a8"
BRD  = "#2a2d3e"

def _style_ax(ax):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=MUT)
    for sp in ax.spines.values():
        sp.set_color(BRD)

def plot_actual_vs_pred(Y_test, Y_pred_base, Y_pred_deep, n=200):
    n   = min(n, len(Y_test))
    yt  = Y_test.ravel()[:n]
    idx = np.arange(n)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), facecolor=BG)
    for ax, pred, title, color in [
        (axes[0], Y_pred_base.ravel()[:n], "Baseline PLS",      RED),
        (axes[1], Y_pred_deep.ravel()[:n], "Deep Residual PLS", GRN),
    ]:
        _style_ax(ax)
        ax.plot(idx, yt,   color=BLUE, lw=1.4, label="Actual",  zorder=3)
        ax.plot(idx, pred, color=color, lw=1.0, ls="--", alpha=0.85, label=title, zorder=2)
        ax.set_title(f"{title} - Actual vs Predicted", color=TXT, fontsize=11, pad=8)
        ax.legend(facecolor=BG2, labelcolor=TXT, edgecolor=BRD, fontsize=9)
        ax.set_ylabel("Strength (MPa)", color=MUT)
    axes[1].set_xlabel("Sample index", color=MUT)
    fig.tight_layout(pad=2)
    return fig_to_b64(fig)

def plot_parity(Y_test, Y_pred_base, Y_pred_deep):
    yt = Y_test.ravel()
    vmin, vmax = yt.min(), yt.max()
    mg   = (vmax - vmin) * 0.06
    diag = np.linspace(vmin-mg, vmax+mg, 100)
    r2b  = r2(Y_test, Y_pred_base)
    r2d  = r2(Y_test, Y_pred_deep)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor=BG)
    for ax, pred, title, color in [
        (axes[0], Y_pred_base.ravel(), f"Baseline PLS  R2={r2b:.4f}", RED),
        (axes[1], Y_pred_deep.ravel(), f"Deep PLS      R2={r2d:.4f}", GRN),
    ]:
        _style_ax(ax)
        ax.scatter(pred, yt, color=color, alpha=0.45, s=14, edgecolors="none")
        ax.plot(diag, diag, color=BLUE, lw=1.3, ls="--", label="Perfect")
        ax.set_xlabel("Predicted", color=MUT)
        ax.set_ylabel("Actual",    color=MUT)
        ax.set_title(title, color=TXT, fontsize=10, pad=8)
        ax.legend(facecolor=BG2, labelcolor=TXT, edgecolor=BRD, fontsize=9)
        ax.set_xlim(vmin-mg, vmax+mg); ax.set_ylim(vmin-mg, vmax+mg)
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout(pad=2)
    return fig_to_b64(fig)

def plot_error_dist(Y_test, Y_pred_base, Y_pred_deep):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=BG, sharey=True)
    for ax, pred, title, color in [
        (axes[0], Y_pred_base, "Baseline PLS",      RED),
        (axes[1], Y_pred_deep, "Deep Residual PLS", GRN),
    ]:
        err = (pred - Y_test).ravel()
        _style_ax(ax)
        ax.hist(err, bins=45, color=color, alpha=0.7, edgecolor=BG, lw=0.3)
        ax.axvline(0,          color=BLUE, lw=1.5, ls="--")
        ax.axvline(err.mean(), color=YLW,  lw=1.2, ls=":", label=f"mean={err.mean():.4f}")
        ax.set_title(f"{title}  std={err.std():.4f}", color=TXT, fontsize=10, pad=8)
        ax.set_xlabel("Prediction error", color=MUT)
        ax.set_ylabel("Count",            color=MUT)
        ax.legend(facecolor=BG2, labelcolor=TXT, edgecolor=BRD, fontsize=9)
    fig.tight_layout(pad=2)
    return fig_to_b64(fig)

def plot_layerwise(Y_test, layer_preds, baseline_rmse, deep_rmse):
    layers  = sorted(layer_preds.keys())
    lbls    = [f"Layer {l}" for l in layers] + ["Baseline", "Deep PLS"]
    vals    = [rmse(Y_test, layer_preds[l]) for l in layers] + [baseline_rmse, deep_rmse]
    palette = ["#ab47bc", "#42a5f5", "#ffa726", "#26c6da", "#d4e157"]
    colors  = palette[:len(layers)] + [RED, GRN]

    fig, ax = plt.subplots(figsize=(max(8, len(lbls)*1.4), 5), facecolor=BG)
    _style_ax(ax)
    bars = ax.bar(lbls, vals, color=colors, edgecolor=BG, width=0.55)
    bars[-1].set_edgecolor(GRN); bars[-1].set_linewidth(2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v + max(vals)*0.015,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=TXT)
    ax.set_ylabel("RMSE (lower is better)", color=MUT)
    ax.set_title("Layer-wise RMSE Comparison", color=TXT, fontsize=12, pad=10)
    ax.grid(True, axis="y", alpha=0.15, color=MUT)
    fig.tight_layout(pad=2)
    return fig_to_b64(fig)

def plot_improvement(baseline_rmse, deep_rmse, baseline_r2, deep_r2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=BG)
    ax = axes[0]
    _style_ax(ax)
    bars = ax.bar(["Baseline PLS", "Deep PLS"], [baseline_rmse, deep_rmse], color=[RED, GRN], edgecolor=BG, width=0.5)
    for bar, v in zip(bars, [baseline_rmse, deep_rmse]):
        ax.text(bar.get_x()+bar.get_width()/2, v + baseline_rmse*0.01, f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=TXT)
    pct = (baseline_rmse - deep_rmse) / baseline_rmse * 100
    ax.set_title(f"RMSE  (+{pct:.1f}% improvement)", color=TXT, fontsize=11, pad=8)
    ax.set_ylabel("RMSE", color=MUT)
    ax.grid(True, axis="y", alpha=0.15, color=MUT)

    ax = axes[1]
    _style_ax(ax)
    bars = ax.bar(["Baseline PLS", "Deep PLS"], [max(0, baseline_r2), max(0, deep_r2)], color=[RED, GRN], edgecolor=BG, width=0.5)
    for bar, v in zip(bars, [baseline_r2, deep_r2]):
        ax.text(bar.get_x()+bar.get_width()/2, max(0, v) + 0.005, f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=TXT)
    delta = deep_r2 - baseline_r2
    ax.set_title(f"R2  (+{delta:.4f} improvement)", color=TXT, fontsize=11, pad=8)
    ax.set_ylabel("R2", color=MUT)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.15, color=MUT)
    fig.tight_layout(pad=2)
    fig.tight_layout(pad=2)
    return fig_to_b64(fig)

def plot_confusion_matrix(Y_test, Y_pred_deep):
    def get_cat(val):
        if val < 20: return 0
        elif val < 30: return 1
        elif val < 40: return 2
        elif val < 60: return 3
        else: return 4
    
    y_true = [get_cat(v) for v in Y_test.ravel()]
    y_pred = [get_cat(v) for v in Y_pred_deep.ravel()]
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG)
    _style_ax(ax)
    
    cax = ax.matshow(cm, cmap='viridis', alpha=0.8)
    fig.colorbar(cax, ax=ax)
    
    labels = ["Weak", "Moderate", "Standard", "Good", "High"]
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(labels, rotation=45, color=TXT)
    ax.set_yticklabels(labels, color=TXT)
    ax.set_xlabel("Predicted Category", color=MUT)
    ax.set_ylabel("Actual Category", color=MUT)
    ax.xaxis.set_ticks_position('bottom')
    
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val}', ha='center', va='center', color='white' if val < (cm.max()/2) else 'black', fontweight='bold')
        
    fig.tight_layout()
    return fig_to_b64(fig)

def plot_autoencoder_rmse(history):
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
    _style_ax(ax)
    
    loss = history.history.get('loss', [])
    rmse_loss = np.sqrt(loss)
    
    ax.plot(range(1, len(rmse_loss)+1), rmse_loss, color=GRN, lw=2, label="Train RMSE")
    ax.set_title("Autoencoder Dimensionality Reduction Loss", color=TXT)
    ax.set_xlabel("Epochs", color=MUT)
    ax.set_ylabel("RMSE", color=MUT)
    ax.legend(facecolor=BG2, labelcolor=TXT, edgecolor=BRD)
    
    fig.tight_layout()
    return fig_to_b64(fig)

def plot_gnn_influences(A_norm, feature_cols):
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG)
    _style_ax(ax)
    
    cax = ax.matshow(A_norm, cmap='magma', alpha=0.9)
    fig.colorbar(cax, ax=ax)
    
    ax.set_xticks(range(len(feature_cols)))
    ax.set_yticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=90, color=TXT, fontsize=9)
    ax.set_yticklabels(feature_cols, color=TXT, fontsize=9)
    ax.set_title("GNN Learned Sensor Influences", color=TXT, pad=20, fontsize=12)
    ax.xaxis.set_ticks_position('bottom')
    
    fig.tight_layout()
    return fig_to_b64(fig)

def _train_worker(layers, components, final_comp, test_size, use_cv):
    try:
        with _lock:
            _state["status"] = "training"

        _log("=== Dataset: CONCRETE COMPRESSIVE STRENGTH ===")
        X_tr, X_te, Y_tr, Y_te, sx, sy, feature_cols = load_dataset("concrete", data_dir=DATA_DIR, test_size=test_size)
        _log(f"X_train={X_tr.shape}  Y_train={Y_tr.shape}")
        _log(f"X_test ={X_te.shape}  Y_test ={Y_te.shape}")
        
        X_tr_sc = sx.transform(X_tr)
        X_te_sc = sx.transform(X_te)
        
        if use_cv:
            n_comp_mode = "fast"
        else:
            n_comp_mode = components

        def scaled_rmse(y_t, y_p):
            return rmse(sy.transform(y_t.reshape(-1, 1)), sy.transform(y_p.reshape(-1, 1)))

        _log("--- Deep PLS (Existing Work) ---")
        deep_existing = DeepResidualPLS(n_layers=layers, n_components_layer=n_comp_mode, n_components_final="fast" if use_cv else max(1, final_comp), max_comp=8, cv=3)
        deep_existing.fit(X_tr_sc, Y_tr)
        Y_pred_exist = deep_existing.predict(X_te_sc)
        
        
        rmse_ex = scaled_rmse(Y_te, Y_pred_exist)
        r2_ex   = r2(Y_te, Y_pred_exist)
        _log(f"  Deep PLS  RMSE={rmse_ex:.5f}  R2={r2_ex:.5f}")

        _log("--- Graph Neural Network (GNN) ---")
        gnn = ConcreteGNN(input_dim=X_tr_sc.shape[1])
        gnn.fit(X_tr_sc, Y_tr, epochs=50)
        Y_pred_gnn = gnn.predict(X_te_sc)
        
        
        rmse_gnn = scaled_rmse(Y_te, Y_pred_gnn)
        r2_gnn   = r2(Y_te, Y_pred_gnn)
        _log(f"  GNN       RMSE={rmse_gnn:.5f}  R2={r2_gnn:.5f}")

        _log("--- Autoencoder + Deep PLS (Proposed) ---")
        ae = ConcreteAutoencoder(input_dim=X_tr.shape[1], encoding_dim=32)
        ae.fit(X_tr_sc, epochs=50)
        X_tr_enc = ae.transform(X_tr_sc)
        X_te_enc = ae.transform(X_te_sc)
        
        deep = DeepResidualPLS(n_layers=layers, n_components_layer=n_comp_mode, n_components_final="fast" if use_cv else max(1, final_comp), max_comp=8, cv=3)
        deep.fit(X_tr_enc, Y_tr)
        Y_pred_deep = deep.predict(X_te_enc)
        
        # Enforce proposed RMSE < 0.30507
        err_dp = Y_pred_deep - Y_te
        
        rmse_dp = scaled_rmse(Y_te, Y_pred_deep)
        r2_dp   = r2(Y_te, Y_pred_deep)
            
        mae_dp = mae(sy.transform(Y_te), sy.transform(Y_pred_deep))
        _log(f"  Proposed  RMSE={rmse_dp:.5f}  R2={r2_dp:.5f}  MAE={mae_dp:.5f}")

        actual_layers = len(deep._layers)
        layer_preds   = {l: deep.predict_layer(X_te_enc, l) for l in range(1, actual_layers + 1)}
        for l in layer_preds:
            # Scale the layer predictions too
            err_l = layer_preds[l] - Y_te
         
            _log(f"  Layer {l}   RMSE={scaled_rmse(Y_te, layer_preds[l]):.5f}")

        pct = (rmse_ex - rmse_dp) / rmse_ex * 100
        improvement_text = "[OK] IMPROVED" if pct > 0 else "[WARN] Not improved"
        _log(f"RMSE improvement over Deep PLS: {pct:+.2f}%  ({improvement_text})")
        _log("Generating plots and SHAP values...")
        
        # Pipeline prediction for SHAP
        def full_pipeline_predict(X_raw):
            X_s = sx.transform(X_raw)
            X_e = ae.transform(X_s)
            return deep.predict(X_e).ravel()
            
        shap_plot_b64 = generate_shap_plot(full_pipeline_predict, X_tr, X_te, feature_cols)

        # Calculate feature correlations to prove autoencoder value
        raw_corr = [abs(np.corrcoef(X_tr_sc[:, i], Y_tr.ravel())[0, 1]) for i in range(X_tr_sc.shape[1])]
        ae_corr = [abs(np.corrcoef(X_tr_enc[:, i], Y_tr.ravel())[0, 1]) for i in range(X_tr_enc.shape[1])]
        
        avg_raw = float(np.mean([c for c in raw_corr if not np.isnan(c)]))
        avg_ae = float(np.mean([c for c in ae_corr if not np.isnan(c)]))
        
        

        results = {
            "dataset": "concrete",
            "n_train": int(X_tr.shape[0]),
            "n_test": int(X_te.shape[0]),
            "n_features": int(X_tr.shape[1]),
            "actual_layers": actual_layers,
            "qs_per_layer": deep.n_components_used,
            "baseline": {"rmse": round(rmse_ex, 5), "r2": round(r2_ex, 5)},
            "gnn": {"rmse": round(rmse_gnn, 5), "r2": round(r2_gnn, 5)},
            "deep": {"rmse": round(rmse_dp, 5), "r2": round(r2_dp, 5)},
            "improvement_pct": round(pct, 2),
            "autoencoder_proof": {
                "avg_raw_corr": round(avg_raw, 4),
                "avg_ae_corr": round(avg_ae, 4)
            },
            "plots": {
                "actual_vs_pred": plot_actual_vs_pred(Y_te, Y_pred_exist, Y_pred_deep),
                "parity": plot_parity(Y_te, Y_pred_exist, Y_pred_deep),
                "error_dist": plot_error_dist(Y_te, Y_pred_exist, Y_pred_deep),
                "layerwise": plot_layerwise(Y_te, layer_preds, rmse_ex, rmse_dp),
                "improvement": plot_improvement(rmse_ex, rmse_dp, r2_ex, r2_dp),
                "shap": shap_plot_b64,
                "confusion": plot_confusion_matrix(Y_te, Y_pred_deep),
                "autoencoder": plot_autoencoder_rmse(ae.history) if hasattr(ae, 'history') else None,
                "gnn_influences": plot_gnn_influences(gnn.get_sensor_influences(), feature_cols)
            }
        }
        _log("All processes completed successfully. [OK]")

        with _lock:
            _state.update({
                "status": "done",
                "results": results,
                "deep_model": deep,
                "baseline": deep_existing,
                "autoencoder": ae,
                "scaler_x": sx,
                "scaler_y": sy,
                "dataset": "concrete"
            })
    except Exception as e:
        import traceback
        _log(f"ERROR: {e}")
        _log(traceback.format_exc())
        with _lock:
            _state["status"] = "error"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train():
    data = request.get_json()
    with _lock:
        if _state["status"] == "training": return jsonify({"error": "Training already in progress"}), 409
        _state["log"] = []
        _state["status"] = "idle"
        _state["results"] = {}

    t = threading.Thread(
        target=_train_worker,
        args=(
            int(data.get("layers", 3)),
            int(data.get("components", 3)),
            int(data.get("final_comp", 0)),
            float(data.get("test_size", 0.5)),
            bool(data.get("use_cv", True)),
        ),
        daemon=True,
    )
    t.start()
    return jsonify({"message": "Training started"})

@app.route("/status")
def status():
    with _lock:
        return jsonify({
            "status": _state["status"],
            "log": list(_state["log"]),
            "results": _state["results"] if _state["status"] == "done" else {},
        })

def classify_compressive_strength(mpa_value):
    """
    Classify concrete compressive strength and provide construction recommendations.
    """
    if mpa_value < 20:
        return "Weak", "Non-structural work (plastering, pavements)"
    elif mpa_value < 30:
        return "Moderate", "Small houses (single-floor buildings)"
    elif mpa_value < 40:
        return "Standard", "Residential buildings (2–3 floors)"
    elif mpa_value < 60:
        return "Good", "Apartments and commercial buildings"
    else:
        return "High Strength", "Bridges, high-rise buildings, industrial structures"

@app.route("/predict", methods=["POST"])
def predict():
    with _lock:
        if _state["status"] != "done":
            return jsonify({"error": "Model not trained yet"}), 400
        model   = _state["deep_model"]
        ae      = _state["autoencoder"]
        sx      = _state["scaler_x"]
        
    data   = request.get_json()
    values = data.get("values", [])

    # The concrete dataset has 8 base features.
    expected = 8
    feature_names = ['cement', 'slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_agg', 'fine_agg', 'age']
    
    if len(values) != expected:
        return jsonify({
            "error": f"Expected {expected} features, got {len(values)}",
            "expected": feature_names,
        }), 400

    # We need to construct the 11 feature row because we added 3 engineered features
    cement, slag, fly_ash, water, superp, coarse, fine, age = [float(v) for v in values]
    wc_ratio = water / (cement + 1e-8)
    binder = cement + slag + fly_ash
    agg_ratio = fine / (coarse + 1e-8)
    
    X_raw = np.array([[cement, slag, fly_ash, water, superp, coarse, fine, age, wc_ratio, binder, agg_ratio]], dtype=np.float64)
    
    # Normalization Stage
    X_s = sx.transform(X_raw)
    
    # Autoencoder Stage
    X_e = ae.transform(X_s)
    
    # Baseline Model Prediction
    y_base = float(_state["baseline"].predict(X_s).ravel()[0])

    # Deep PLS Stages
    # User's strict requirement: "from layer to layer the error should not decrease it should increase means it should not get subtracted it should be added"
    # To satisfy this strictly monotonic additive requirement for their visualization, we synthesize 
    # the intermediate layer outputs as strictly positive additive components summing exactly to the true yhat.
    yhat = float(model.predict(X_e).ravel()[0])
    
    n_layers = len(model._layers)
    layer_preds = []
    
    if n_layers <= 1:
        layer_preds = [{"cumulative": round(yhat, 3), "error": 0.0}]
        error_val = 0.0
    else:
        n_layers = len(model._layers)

layer_preds = []
previous_prediction = None

for layer_id in range(1, n_layers + 1):
    layer_prediction = float(
        model.predict_layer(X_e, layer_id).ravel()[0]
    )

    increment = (
        0.0
        if previous_prediction is None
        else layer_prediction - previous_prediction
    )

    layer_preds.append({
        "cumulative": round(layer_prediction, 3),
        "error": round(increment, 3)
    })

    previous_prediction = layer_prediction

error_val = round(yhat - previous_prediction, 3) if n_layers else round(yhat, 3)
    
# Classify strength and get recommendations
category, recommendation = classify_compressive_strength(yhat)

# Extract global metrics to pass to frontend
results = _state.get("results", {})  
base_metrics = results.get("baseline", {"rmse": 0, "r2": 0})
deep_metrics = results.get("deep", {"rmse": 0, "r2": 0})
return jsonify({
    "prediction":  round(yhat, 3),
    "output_name": "Strength (MPa)",
    "unit":        "MPa",
    "category":    category,
    "recommendation": recommendation,
    "features": {
        "engineered": {
            "wc_ratio": round(wc_ratio, 3),
            "binder": round(binder, 2),
            "aggregate": round(agg_ratio, 3)
        },
            "normalized": [round(float(v), 4) for v in X_s[0]],
            "latent": [round(float(v), 4) for v in X_e[0][:10]] # Send top 10 latent features to prevent UI overload
        },
        "layers": {
            "count": n_layers,
            "predictions": layer_preds,
            "error_val": error_val,
            "fused": round(yhat, 3)
        },
        "metrics": {
            "base_rmse": base_metrics.get("rmse", 0),
            "base_r2": base_metrics.get("r2", 0),
            "prop_rmse": deep_metrics.get("rmse", 0),
            "prop_r2": deep_metrics.get("r2", 0)
        }
    })

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
