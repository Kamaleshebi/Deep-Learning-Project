import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

class ConcreteAutoencoder:
    def __init__(self, input_dim, encoding_dim=6):
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(8, activation='relu')(input_layer)
        encoded = Dense(encoding_dim, activation='relu')(encoded)
        
        decoded = Dense(8, activation='relu')(encoded)
        decoded = Dense(input_dim, activation='linear')(decoded)
        
        self.autoencoder = Model(input_layer, decoded)
        self.encoder = Model(input_layer, encoded)
        
        self.autoencoder.compile(optimizer='adam', loss='mse')
        
    def fit(self, X, epochs=100, batch_size=16, verbose=0):
        self.autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=verbose)
        return self
        
    def transform(self, X):
        return self.encoder.predict(X, verbose=0)
