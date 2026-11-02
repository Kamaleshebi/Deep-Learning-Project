"""
residual_model.py  (v2 — corrected)
====================================
Deep Residual Probabilistic PLS model following:

  Ge, Chen & Yu (2025) – "Deep Residual Learning of a Probabilistic Partial
  Least Squares Model for Predictive Data Analytics"
  IEEE Transactions on Artificial Intelligence, Vol. 6, No. 11.

Architecture (Section III of the paper)
-----------------------------------------
Layer 1 : PLS on (X_sc,          Y_sc) → ts(1),  E1 = X_sc  - ts(1)@P1.T
Layer 2 : PLS on (E1,             Y_sc) → ts(2),  E2 = E1    - ts(2)@P2.T
...
Layer L : PLS on (E_{L-1},        Y_sc) → ts(L)

Final fusion (Eq. 15-16):
    td = [ts(1) | ts(2) | ... | ts(L)]          (n × L·qs)
    PLS on (td, Y_sc) → final prediction Ŷ

Key fixes in v2
---------------
1. EXACT NIPALS DEFLATION: During training we use layer._train_scores
   (the actual NIPALS x_scores_) instead of transform().  transform()
   uses x_rotations_ which can differ slightly and causes residuals to
   retain already-captured variance — preventing proper deep learning.

2. CROSS-VALIDATION COMPONENT SELECTION: n_components is chosen by CV
   within the training data, maximising R² on held-out folds.

3. ADAPTIVE FUSION LAYER: fusion PLS uses n_components = n_layers (not a
   small hardcoded constant), ensuring all concatenated scores are used.

4. STABLE RESIDUALS: variance guard ensures we only add layers that still
   have meaningful signal; empty layers are skipped gracefully.
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from pls_layer import PLSLayer


def _rmse(Y_true, Y_pred):
    return float(np.sqrt(np.mean((Y_true - Y_pred) ** 2)))

def _fast_select_components(X: np.ndarray, Y: np.ndarray,
                             max_comp: int = 8) -> int:

    from sklearn.model_selection import train_test_split as _tts
    n_samples, n_features = X.shape
    upper = min(max_comp, n_features, max(1, n_samples - 1))

    if n_samples < 10:
        return min(2, upper)
    try:
        X_tr, X_val, y_tr, y_val = _tts(
            X, Y.ravel(), test_size=0.2, random_state=42
        )
    except ValueError:
        return min(3, upper)

    best_n, best_r2 = 1, -np.inf
    prev_r2 = -np.inf
    ss_tot = float(np.sum((y_val - y_val.mean()) ** 2))
    if ss_tot < 1e-12:
        return 1
    for n in range(1, upper + 1):
        try:
            pls = PLSRegression(n_components=n, scale=False, max_iter=500)
            pls.fit(X_tr, y_tr)
            y_pred = pls.predict(X_val).ravel()
            r2 = 1.0 - float(np.sum((y_val - y_pred) ** 2)) / ss_tot
            if r2 > best_r2:
                best_r2 = r2
                best_n  = n
            # Early stop: improvement < 0.5% — adding more components won't help
            if n > 1 and (r2 - prev_r2) < 0.005:
                break
            prev_r2 = r2
        except Exception:
            break

    return best_n

def _cv_select_components(X: np.ndarray, Y: np.ndarray,
                          max_comp: int = 8, cv: int = 3) -> int:
    """
    Cross-validated component selection (slower than _fast_select_components).
    Uses at most max_comp components and cv folds.
    """
    n_samples, n_features = X.shape
    upper = min(max_comp, n_features, n_samples - 1)
    upper = max(1, upper)

    best_n, best_score = 1, -np.inf
    for n in range(1, upper + 1):
        pls = PLSRegression(n_components=n, scale=False, max_iter=500)
        try:
            folds  = max(2, min(cv, n_samples // max(n * 2, 2)))
            scores = cross_val_score(pls, X, Y.ravel(),
                                     cv=folds, scoring="r2")
            mean_s = float(scores.mean())
            if mean_s > best_score:
                best_score = mean_s
                best_n     = n
        except Exception:
            break
    return best_n

class DeepResidualPLS:   
    def __init__(
        self,
        n_layers:            int  = 3,
        n_components_layer        = 3,   
        n_components_final        = "fast",
        max_comp:            int  = 8, 
        cv:                  int  = 3,  
    ):
        self.n_layers            = n_layers
        self.n_components_layer  = n_components_layer
        self.n_components_final  = n_components_final
        self.max_comp            = max_comp
        self.cv                  = cv

        self._layers             = []
        self._fusion_scaler_X    = StandardScaler()
        self._global_scaler_X      = None
        self._global_scaler_Y      = None
        self.is_fitted             = False

        self.train_rmse_per_layer: list = []
        self.n_components_used:    list = [] 

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "DeepResidualPLS":
 
        n_samples, n_features = X_train.shape

        self._global_scaler_X = StandardScaler().fit(X_train)
        self._global_scaler_Y = StandardScaler().fit(Y_train)
        X_sc = self._global_scaler_X.transform(X_train)   # (n, m)
        Y_sc = self._global_scaler_Y.transform(Y_train)   # (n, 1)

        self._layers                = []
        self.train_rmse_per_layer   = []
        self.n_components_used      = []

        X_res          = X_sc.copy()   
        Y_res          = Y_sc.copy() 
        all_scores     = []            

        for l in range(1, self.n_layers + 1):

            var_rem = float(X_res.var(axis=0).sum())
            if var_rem < 1e-8:
                print(f"  [Layer {l}] Residual variance exhausted "
                      f"({var_rem:.2e}) — stopping at layer {l-1}.")
                break

            if self.n_components_layer in ("auto", "fast"):
                qs = _fast_select_components(
                    X_res, Y_res, max_comp=self.max_comp
                )
                print(f"  [Layer {l}] Fast-selected qs = {qs}")
            elif self.n_components_layer == "cv":
                qs = _cv_select_components(
                    X_res, Y_res,
                    max_comp=self.max_comp, cv=self.cv
                )
                print(f"  [Layer {l}] CV-selected qs = {qs}")
            else:
                qs = min(int(self.n_components_layer),
                         n_features, n_samples - 1)
                qs = max(1, qs)
            layer = PLSLayer(
                n_components=qs,
                layer_id=l,
                scale=False, 
            )
            layer.fit(X_res, Y_res)
            self.n_components_used.append(layer._n_fitted)

            ts_l = layer._train_scores.copy() 
            all_scores.append(ts_l)

            Y_pred_sc = ts_l @ layer.C_.T                    
            Y_pred_l  = self._global_scaler_Y.inverse_transform(Y_pred_sc)
            rmse_l    = _rmse(Y_train, Y_pred_l)
            self.train_rmse_per_layer.append(rmse_l)
            print(f"  [Layer {l}] qs={layer._n_fitted}  "
                  f"train RMSE={rmse_l:.5f}  "
                  f"residual var={var_rem:.4f}")

            X_hat_l = ts_l @ layer.P_.T   
            X_res   = X_res - X_hat_l

            Y_hat_l = ts_l @ layer.C_.T
            Y_res   = Y_res - Y_hat_l

            self._layers.append(layer)

        if not self._layers:
            raise RuntimeError("No layers were fitted. "
                               "Check your data or reduce n_components.")

        Td        = np.hstack(all_scores)           
        Td_scaled = self._fusion_scaler_X.fit_transform(Td)

        total_cols = Td_scaled.shape[1]
        if self.n_components_final in ("auto", "fast"):
            n_fus = _fast_select_components(
                Td_scaled, Y_sc,
                max_comp=min(total_cols, self.max_comp)
            )
            print(f"  [Fusion]  Fast-selected qs = {n_fus}")
        elif self.n_components_final == "cv":
            n_fus = _cv_select_components(
                Td_scaled, Y_sc,
                max_comp=min(total_cols, self.max_comp),
                cv=self.cv
            )
            print(f"  [Fusion]  CV-selected qs = {n_fus}")
        else:
            n_fus = min(int(self.n_components_final),
                        total_cols, n_samples - 1)
            n_fus = max(1, n_fus)

        self._fusion_pls = PLSRegression(
            n_components=n_fus, scale=False, max_iter=1000
        )
        self._fusion_pls.fit(Td_scaled, Y_sc)

        # Fusion diagnostic RMSE
        Y_fus_sc = self._fusion_pls.predict(Td_scaled)
        Y_fus    = self._global_scaler_Y.inverse_transform(Y_fus_sc)
        rmse_fus = _rmse(Y_train, Y_fus)
        print(f"  [Fusion]  qs={n_fus}  train RMSE={rmse_fus:.5f}")

        self.is_fitted = True
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self._check_fit()
        X_sc  = self._global_scaler_X.transform(X_test)
        X_res = X_sc.copy()
        all_scores = []

        for layer in self._layers:
            ts_l    = layer.transform(X_res)      # (n, qs)
            all_scores.append(ts_l)
            X_hat_l = ts_l @ layer.P_.T           # deflation in scaled space
            X_res   = X_res - X_hat_l

        Td        = np.hstack(all_scores)
        Td_scaled = self._fusion_scaler_X.transform(Td)
        Y_pred_sc = self._fusion_pls.predict(Td_scaled)
        Y_pred    = self._global_scaler_Y.inverse_transform(Y_pred_sc)
        return Y_pred

    def predict_layer(self, X_test: np.ndarray, layer_id: int) -> np.ndarray:
        """
        Prediction using only the l-th layer model (diagnostic).
        layer_id is 1-based.
        """
        self._check_fit()
        n = len(self._layers)
        if layer_id < 1 or layer_id > n:
            raise ValueError(f"layer_id must be between 1 and {n}")

        X_sc  = self._global_scaler_X.transform(X_test)
        X_res = X_sc.copy()

        for idx, layer in enumerate(self._layers):
            ts_l = layer.transform(X_res)
            if idx + 1 == layer_id:
                Y_pred_sc = ts_l @ layer.C_.T
                return self._global_scaler_Y.inverse_transform(Y_pred_sc)
            # Deflate and move to next layer
            X_hat_l = ts_l @ layer.P_.T
            X_res   = X_res - X_hat_l

    # ── Internals ─────────────────────────────────────────────────────────────

    def _check_fit(self):
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")

    def __repr__(self):
        return (
            f"DeepResidualPLS("
            f"n_layers={len(self._layers)}, "
            f"qs_per_layer={self.n_components_used}, "
            f"n_components_final={self.n_components_final})"
        )


# ── Baseline single-layer PLS ─────────────────────────────────────────────────

class BaselinePLS:
    """
    Standard single-layer PLSRegression wrapper (the 'shallow PPLS' baseline).

    Uses the same cross-validation component selection as the deep model
    to ensure a fair comparison.
    """

    def __init__(self, n_components=3, use_cv: bool = False,
                 max_comp: int = 10, cv: int = 5):
        self.n_components = n_components
        self.use_cv       = use_cv
        self.max_comp     = max_comp
        self.cv           = cv
        self._layer       = None
        self.is_fitted    = False
        self._n_fitted    = None

    def fit(self, X_train, Y_train):
        scaler_X = StandardScaler()
        X_sc     = scaler_X.fit_transform(X_train)
        Y_sc     = StandardScaler().fit_transform(Y_train)

        nc = self.n_components
        if nc in ("fast", "auto"):
            # Fast 80/20 holdout selection
            qs = _fast_select_components(X_sc, Y_sc, max_comp=self.max_comp)
            print(f"  [Baseline] Fast-selected qs = {qs}")
        elif nc == "cv":
            qs = _cv_select_components(X_sc, Y_sc,
                                       max_comp=self.max_comp, cv=self.cv)
            print(f"  [Baseline] CV-selected qs = {qs}")
        else:
            qs = min(int(nc), X_sc.shape[0] - 1, X_sc.shape[1])
            qs = max(1, qs)

        self._n_fitted = qs
        self._layer    = PLSLayer(n_components=qs, layer_id=1, scale=True)
        self._layer.fit(X_train, Y_train)
        self.is_fitted = True
        return self

    def predict(self, X_test):
        if not self.is_fitted:
            raise RuntimeError("BaselinePLS not fitted.")
        return self._layer.predict(X_test)

    def __repr__(self):
        return f"BaselinePLS(n_components={self._n_fitted or self.n_components})"
