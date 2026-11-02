import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64

class ConcreteAutoencoder:
    def __init__(self, input_dim, encoding_dim=8):
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(32, activation='relu')(input_layer)
        encoded = Dense(encoding_dim, activation='linear')(encoded)
        
        decoded = Dense(32, activation='relu')(encoded)
        decoded = Dense(input_dim, activation='linear')(decoded)
        
        self.autoencoder = Model(input_layer, decoded)
        self.encoder = Model(input_layer, encoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')
        
    def fit(self, X, epochs=100, batch_size=16, verbose=0):
        self.history = self.autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=verbose)
        return self
        
    def transform(self, X):
        return self.encoder.predict(X, verbose=0)

class AutoencoderPredictor:
    def __init__(self, input_dim):
        inputs = Input(shape=(input_dim,))
        encoded = Dense(16, activation='relu')(inputs)
        encoded = Dense(8, activation='relu')(encoded)
        outputs = Dense(1, activation='linear')(encoded)
        
        self.model = Model(inputs, outputs)
        self.model.compile(optimizer='adam', loss='mse')
        
    def fit(self, X, y, epochs=100, batch_size=16, verbose=0):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        return self
        
    def predict(self, X):
        return self.model.predict(X, verbose=0)

def generate_shap_plot(full_pipeline_predict_func, X_train, X_test, feature_names):
    """
    Generates a SHAP summary plot using the full predictive pipeline.
    full_pipeline_predict_func: function that takes raw features X and outputs predictions.
    """
    # Background dataset for SHAP to integrate over (reduced for speed)
    background = X_train[np.random.choice(X_train.shape[0], min(15, X_train.shape[0]), replace=False)]
    
    explainer = shap.KernelExplainer(full_pipeline_predict_func, background)
    
    # Calculate SHAP values for a tiny subset of the test set to save computation time
    X_test_sample = X_test[:15]
    shap_values = explainer.shap_values(X_test_sample)
    
    # KernelExplainer returns a list of arrays for multi-class or single array for regression.
    # Handle both cases:
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        
    fig = plt.figure(figsize=(10, 6), facecolor='#ffffff')
    ax = fig.gca()
    ax.set_facecolor('#ffffff')
    ax.tick_params(colors='#333333')
    
    # Set text colors inside the SHAP plot
    plt.rcParams['text.color'] = '#333333'
    plt.rcParams['axes.labelcolor'] = '#333333'
    
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False)
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64
