"""
pls_layer.py
============
A single probabilistic-PLS-inspired layer.

The paper builds a generative model:
    x = P·ts + Q·tu + ex
    y = C·ts + ey

where ts = supervised latent variables (relate X to Y)
      tu = unsupervised latent variables (explain X only)

For practical computation we use sklearn's PLSRegression which extracts
exactly these "supervised" scores via the NIPALS algorithm.

KEY FIX (v2):
    The residual deflation MUST use the actual NIPALS training scores
    (x_scores_), NOT the transform() approximation.  Using transform()
    introduces a small approximation error that compounds across layers
    and prevents proper residual learning.

    Reconstruction:  X̂ = T @ P.T   (T = x_scores_, P = x_loadings_)
    Residual:        E  = X - X̂    (exact orthogonal deflation)
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


class PLSLayer:
    """
    A single PLS layer that:
      1. Fits PLS on input residuals and Y
      2. Extracts supervised latent variables (scores)
      3. Reconstructs X from those scores (for next-layer residuals)
      4. Predicts Y from scores (for evaluation)

    Parameters
    ----------
    n_components : int or 'auto'
        Number of supervised latent variables (qs in paper).
        'auto' → cross-validate 1..max_comp to pick the best.
    layer_id : int
        Human-readable layer index (1-based).
    scale : bool
        Whether to internally standardise X before PLS (recommended when
        data has NOT been pre-scaled globally).
    max_comp : int
        Upper bound when n_components='auto'.
    cv : int
        Folds used for cross-validation when n_components='auto'.
    """

    def __init__(self, n_components=3, layer_id: int = 1,
                 scale: bool = True, max_comp: int = 10, cv: int = 5):
        self.n_components = n_components
        self.layer_id     = layer_id
        self.scale        = scale
        self.max_comp     = max_comp
        self.cv           = cv

        self._scaler_X = StandardScaler() if scale else None
        self._scaler_Y = StandardScaler() if scale else None

        # Set after fit()
        self.P_            = None   # X-loadings  (m × qs)
        self.C_            = None   # Y-loadings  (r × qs)
        self._pls          = None
        self._n_fitted     = None   # actual n_components used
        self.is_fitted     = False

        # Exact training scores from NIPALS (used for residual deflation)
        self._train_scores = None   # (n_train, qs)

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, X_residual: np.ndarray, Y: np.ndarray) -> "PLSLayer":
        """
        Fit PLS on residual input and target output.

        Parameters
        ----------
        X_residual : (n, m) array  – input (or its residual from previous layer)
        Y          : (n, r) array  – target output
        """
        Xf = self._scale_X(X_residual, fit=True)
        Yf = self._scale_Y(Y,          fit=True)

        # ── Select n_components ───────────────────────────────────────────
        if self.n_components == "auto":
            n_comp = self._cv_select(Xf, Yf)
        else:
            n_comp = min(int(self.n_components),
                         Xf.shape[0] - 1,
                         Xf.shape[1])
            n_comp = max(1, n_comp)

        self._n_fitted = n_comp
        self._pls = PLSRegression(n_components=n_comp,
                                  scale=False, max_iter=1000)
        self._pls.fit(Xf, Yf)

        # ── Loading matrices ──────────────────────────────────────────────
        self.P_ = self._pls.x_loadings_    # (m, qs)
        self.C_ = self._pls.y_loadings_    # (r, qs)

        # ── CRITICAL: store exact NIPALS scores (NOT transform()) ─────────
        # These are used for exact orthogonal deflation in the residual step.
        # transform() uses x_rotations_ which gives a slightly different result,
        # causing the residual to retain information already captured by this layer.
        self._train_scores = self._pls.x_scores_   # (n, qs)

        self.is_fitted = True
        return self

    # ── Transform (for inference / test data) ─────────────────────────────────

    def transform(self, X_residual: np.ndarray) -> np.ndarray:
        """
        Extract supervised latent scores ts from residual input.
        (Uses x_rotations_ — appropriate for new/test data.)

        Returns
        -------
        ts : (n, qs) ndarray
        """
        self._check_fit()
        Xf     = self._scale_X(X_residual, fit=False)
        result = self._pls.transform(Xf)
        ts     = result[0] if isinstance(result, tuple) else result
        return ts

    # ── Reconstruct X from TRAINING scores (exact deflation) ─────────────────

    def get_train_scores(self) -> np.ndarray:
        """Return the exact NIPALS training scores (n_train, qs)."""
        self._check_fit()
        return self._train_scores

    # ── Reconstruct X from scores ─────────────────────────────────────────────

    def reconstruct(self, X_residual: np.ndarray) -> np.ndarray:
        """
        Reconstruct X̂ = ts · P.T in the original (un-scaled) space.
        Used for computing residuals on NEW data (test / inference).

        For TRAINING data, use get_train_scores() and do:
            X_hat = train_scores @ P_.T   (in scaled space if scale=False)

        Returns
        -------
        X_reconstructed : (n, m) ndarray  in original input scale
        """
        self._check_fit()
        Xf     = self._scale_X(X_residual, fit=False)
        result = self._pls.transform(Xf)
        ts     = result[0] if isinstance(result, tuple) else result

        # Reconstruct in scaled space
        X_hat_scaled = ts @ self.P_.T     # (n, m)

        # Invert scaling
        if self._scaler_X is not None:
            X_hat = self._scaler_X.inverse_transform(X_hat_scaled)
        else:
            X_hat = X_hat_scaled

        return X_hat

    # ── Predict Y ─────────────────────────────────────────────────────────────

    def predict(self, X_residual: np.ndarray) -> np.ndarray:
        """
        Direct layer prediction: Ŷ from scores.

        Returns
        -------
        Y_pred : (n, r) ndarray in *original* Y scale
        """
        self._check_fit()
        Xf        = self._scale_X(X_residual, fit=False)
        Y_hat_sc  = self._pls.predict(Xf)   # (n, r)

        if self._scaler_Y is not None:
            Y_hat = self._scaler_Y.inverse_transform(Y_hat_sc)
        else:
            Y_hat = Y_hat_sc

        return Y_hat

    # ── Cross-validation component selection ──────────────────────────────────

    def _cv_select(self, Xf: np.ndarray, Yf: np.ndarray) -> int:
        """
        Use cross-validated R² to select the optimal n_components.
        """
        n_samples, n_features = Xf.shape
        upper = min(self.max_comp, n_features, n_samples - 1)
        upper = max(1, upper)

        best_n, best_score = 1, -np.inf
        for n in range(1, upper + 1):
            pls = PLSRegression(n_components=n, scale=False, max_iter=1000)
            try:
                folds  = min(self.cv, n_samples // max(n, 1))
                folds  = max(2, folds)
                scores = cross_val_score(
                    pls, Xf, Yf,
                    cv=folds, scoring="r2"
                )
                mean_s = float(scores.mean())
                if mean_s > best_score:
                    best_score = mean_s
                    best_n     = n
            except Exception:
                break
        return best_n

    # ── Internals ─────────────────────────────────────────────────────────────

    def _scale_X(self, X, fit=False):
        if self._scaler_X is None:
            return X
        if fit:
            return self._scaler_X.fit_transform(X)
        return self._scaler_X.transform(X)

    def _scale_Y(self, Y, fit=False):
        if self._scaler_Y is None:
            return Y
        if fit:
            return self._scaler_Y.fit_transform(Y)
        return self._scaler_Y.transform(Y)

    def _check_fit(self):
        if not self.is_fitted:
            raise RuntimeError(f"PLSLayer {self.layer_id} has not been fitted yet.")

    def __repr__(self):
        return (f"PLSLayer(layer={self.layer_id}, "
                f"n_components={self._n_fitted or self.n_components}, "
                f"fitted={self.is_fitted})")
