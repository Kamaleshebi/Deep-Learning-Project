import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model

class SensorGCN(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.num_sensors = input_shape[-1]
        # Learnable adjacency matrix representing influences between sensors
        self.A_raw = self.add_weight(shape=(self.num_sensors, self.num_sensors), 
                                     initializer='glorot_uniform', trainable=True, name='sensor_adj')
        
    def call(self, X):
        # Enforce non-negativity and add self-loops
        A = tf.nn.relu(self.A_raw) + tf.eye(self.num_sensors)
        # Row normalize the adjacency matrix
        row_sum = tf.reduce_sum(A, axis=1, keepdims=True)
        A_norm = A / (row_sum + 1e-8)
        
        # Propagate signal across the sensor graph: (batch, num_sensors) x (num_sensors, num_sensors)
        output = tf.matmul(X, A_norm)
        return output

class ConcreteGNN:
    def __init__(self, input_dim):
        inputs = Input(shape=(input_dim,))
        
        # Sensor Graph Layer
        gcn = SensorGCN()(inputs)
        
        # Regression head
        dense1 = Dense(16, activation='relu')(gcn)
        dense2 = Dense(8, activation='relu')(dense1)
        outputs = Dense(1, activation='linear')(dense2)
        
        self.model = Model(inputs, outputs)
        self.model.compile(optimizer='adam', loss='mse')
        
    def fit(self, X, y, epochs=150, batch_size=32, verbose=0):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        return self
        
    def predict(self, X):
        return self.model.predict(X, verbose=0)
        
    def get_sensor_influences(self):
        # Extract the learned adjacency matrix
        layer = self.model.layers[1]
        A_raw = layer.A_raw.numpy()
        A = np.maximum(A_raw, 0) + np.eye(A_raw.shape[0])
        row_sum = A.sum(axis=1, keepdims=True)
        return A / (row_sum + 1e-8)
