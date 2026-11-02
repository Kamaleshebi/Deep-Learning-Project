import numpy as np
from sklearn.metrics import r2_score

def rmse(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((Y_true - Y_pred) ** 2)))

def mae(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(Y_true - Y_pred)))

def r2(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    return float(r2_score(Y_true.ravel(), Y_pred.ravel()))

def mape(Y_true: np.ndarray, Y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((Y_true - Y_pred) / (np.abs(Y_true) + eps))) * 100)

def evaluate_model(model_name: str, Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    results = {
        "model": model_name,
        "rmse":  rmse(Y_true, Y_pred),
        "mae":   mae(Y_true, Y_pred),
        "r2":    r2(Y_true, Y_pred),
        "mape":  mape(Y_true, Y_pred),
    }
    print(f"\n{'─'*50}")
    print(f"  Model : {model_name}")
    print(f"  RMSE  : {results['rmse']:.5f}")
    print(f"  MAE   : {results['mae']:.5f}")
    print(f"  R²    : {results['r2']:.5f}")
    print(f"  MAPE  : {results['mape']:.2f}%")
    print(f"{'─'*50}")
    return results

def compare_models(results_list: list[dict]) -> None:
    header = f"{'Model':<30} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'MAPE%':>10}"
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON")
    print("=" * 70)
    print(header)
    print("-" * 70)
    for r in results_list:
        print(
            f"{r['model']:<30} "
            f"{r['rmse']:>10.5f} "
            f"{r['mae']:>10.5f} "
            f"{r['r2']:>10.5f} "
            f"{r['mape']:>10.2f}"
        )
    print("=" * 70)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Y_true = rng.standard_normal((100, 1))
    Y_pred = Y_true + rng.standard_normal((100, 1)) * 0.1
    r = evaluate_model("Demo Model", Y_true, Y_pred)
    print(r)
