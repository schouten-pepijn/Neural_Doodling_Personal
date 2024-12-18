import os
os.environ["KERAS_BACKEND"] = "tensorflow" # tesorflow, jax, torch
import keras
import keras.layers as layers
from keras import ops

import pandas as pd

class Sampling(layers.Layer):
    """Use z_mean and z_log_var to sample z, the vector encoding a digit"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.seed_gen = keras.random.SeedGenerator(87)
        
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        eps = keras.random.normal(shape=(batch, dim),
                                  seed=self.seed_gen)
        z = z_mean + ops.exp(0.5 * z_log_var) * eps
        return z


class Encoder(layers.Layer):
    """maps digits to a triplet (z_mean, z_log_var, z)"""
    def __init__(self, latent_dim=32, intermediate_dim=64,
                 name="encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        
    def build(self, input_shape):
        self.dense_proj = layers.Dense(self.intermediate_dim, activation="relu")
        self.dense_mean = layers.Dense(self.latent_dim)
        self.dense_log_var = layers.Dense(self.latent_dim)
        self.sampling = Sampling()
    
    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling(inputs=(z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, encoded digit vector, into a readable digit"""
    def __init__(self, input_shape,
                 intermediate_dim=64,
                 name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.intermediate_dim = intermediate_dim
        self.input_shape = input_shape
    
    def build(self):
        self.dense_proj = layers.Dense(self.intermediate_dim, activation="relu")
        self.dense_output = layers.Dense(self.input_shape[-1], activation="sigmoid")
        
    def call(self, inputs):
        x = self.dense_proj(inputs)
        outputs = self.dense_output(x)
        return outputs
        

class VariationalAutoEncoder(keras.Model):
    """end-to-end model"""
    def __init__(self,
                 intermediate_dim=64,
                 latent_dim=32,
                 name="autoencoder",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.original_dim = original_dim
        self.intermediate_dim= intermediate_dim
        self.latent_dim = latent_dim
    
    def build(self, input_shape):
        self.encoder = Encoder(latent_dim=self.latent_dim,
                               intermediate_dim=self.intermediate_dim)
        self.decoder = Decoder(intermediate_dim=self.intermediate_dim,
                               input_shape=input_shape)
        
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        
        # custom loss
        kl_loss = -0.5 * ops.mean(
            z_log_var - ops.square(z_mean) - ops.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        
        return reconstructed
    

(x_train, _), _ = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255

original_dim = 784
intermediate_dim = 128
latent_dim = 64
vae = VariationalAutoEncoder(
                             intermediate_dim=intermediate_dim,
                             latent_dim=latent_dim)

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
criterion = keras.losses.MeanSquaredError()
metrics = keras.metrics.MeanAbsoluteError()
vae.compile(loss=criterion, optimizer=optimizer, metrics=[metrics])

history = vae.fit(x_train, x_train, epochs=10,
                  batch_size=16, validation_split=0.2)

results = pd.DataFrame(history.history)

print(results.tail(2))
