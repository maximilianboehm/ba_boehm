import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ReLU
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
import numpy as np

def run_vae_augmentation(X_data, times_to_aug):
    # define dataset
    X_train = X_data
    # number of input columns
    n_inputs = X_train.shape[1]
    # scale data
    t = MinMaxScaler()
    t.fit(X_train)
    X_train = t.transform(X_train)

    # build the encoder
    latent_dim = 2
    encoder_inputs = keras.Input(shape=(n_inputs,))
    e = Dense(n_inputs)(encoder_inputs)
    e = ReLU()(e)
    e = Dense(40, activation='sigmoid')(e)
    z_mean = Dense(latent_dim)(e)
    z_log_var = Dense(latent_dim)(e)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z])


    # build the decoder
    latent_inputs = keras.Input(shape=(latent_dim,))
    d = Dense(n_inputs)(latent_inputs)
    d = ReLU()(d)
    d = Dense(40, activation='sigmoid')(d)
    output = Dense(n_inputs, activation='linear')(d)
    #output = Dense(n_inputs, activation='sigmoid')(d)
    decoder = keras.Model(latent_inputs, output)

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.SGD(lr=1e-3))
    history = vae.fit(X_train, epochs=100, batch_size=128)

    # for plotting loss
    '''
    # plot loss
    pyplot.plot(history.history['loss'], label='train_loss')
    #pyplot.plot(history.history['reconstruction_loss'], label='train_recon')
    #pyplot.plot(history.history['kl_loss'], label='train_kl')
    pyplot.legend()
    pyplot.show()
    '''

    result = []
    for i in range(times_to_aug):
        z_mean, z_log_var, z = vae.encoder.predict(X_train)
        part_res = vae.decoder.predict(z)
        part_res = t.inverse_transform(part_res)
        result.append(part_res)
    result = np.concatenate(result)
    print(result)
    print(result.shape)

    return result


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z (vector)."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), mean=0., stddev=1.)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean()
        self.reconstruction_loss_tracker = keras.metrics.Mean()
        self.kl_loss_tracker = keras.metrics.Mean()

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                    keras.losses.mean_squared_error(data, reconstruction)
            )
            kl_loss = -0.5 * K.mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(kl_loss)
            total_loss = tf.reduce_sum(reconstruction_loss + kl_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }