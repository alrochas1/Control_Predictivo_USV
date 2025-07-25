import tensorflow as tf
import numpy as np

class ControlActivation(tf.keras.layers.Layer):
    def call(self, x):
        v_max = 10

        # v = tf.nn.softplus(x[:, 0]) * v_max
        v = tf.sigmoid(x[:, 0]) * v_max
        phi = tf.tanh(x[:, 1]) * np.pi / 2

        return tf.stack([v, phi], axis=1)