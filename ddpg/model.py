import tensorflow as tf
import numpy as np


def build_models(state_dim, action_dim, layer_dims=[400, 300], upper_bound=1):
    actor = get_actor(state_dim, action_dim, layer_dims, upper_bound)
    critic = get_critic(state_dim, action_dim, layer_dims)
    return actor, critic


def get_actor(state_dim, action_dim, layer_dims, upper_bound):
    inputs = tf.keras.layers.Input(shape=(state_dim,))
    f1 = 1. / np.sqrt(layer_dims[0])
    out = tf.keras.layers.Dense(
            layer_dims[0], activation="relu",
            kernel_initializer=tf.keras.initializers.RandomUniform(-f1, f1),
            bias_initializer=tf.keras.initializers.RandomUniform(-f1, f1)
        )(inputs)

    batch1 = tf.keras.layers.BatchNormalization()(out)
    layer1_activation = tf.nn.relu(batch1)

    f2 = 1. / np.sqrt(layer_dims[1])
    out = tf.keras.layers.Dense(
            layer_dims[1], activation="relu",
            kernel_initializer=tf.keras.initializers.RandomUniform(-f2, f2),
            bias_initializer=tf.keras.initializers.RandomUniform(-f2, f2)
        )(layer1_activation)

    batch2 = tf.keras.layers.BatchNormalization()(out)
    layer2_activation = tf.nn.relu(batch2)

    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    outputs = tf.keras.layers.Dense(
            action_dim,
            activation="tanh",
            kernel_initializer=last_init,
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )(layer2_activation)

    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic(state_dim, action_dim, layer_dims):
    state_input = tf.keras.layers.Input(shape=(state_dim))
    action_input = tf.keras.layers.Input(shape=(action_dim))
    concat = tf.keras.layers.Concatenate()([state_input, action_input])

    f1 = 1. / np.sqrt(layer_dims[0])
    out = tf.keras.layers.Dense(
            layer_dims[0], activation="relu",
            kernel_initializer=tf.keras.initializers.RandomUniform(-f1, f1),
            bias_initializer=tf.keras.initializers.RandomUniform(-f1, f1)
        )(concat)

    batch1 = tf.keras.layers.BatchNormalization()(out)
    layer1_activation = tf.nn.relu(batch1)

    f2 = 1. / np.sqrt(layer_dims[1])
    out = tf.keras.layers.Dense(
            layer_dims[1], activation="relu",
            kernel_initializer=tf.keras.initializers.RandomUniform(-f2, f2),
            bias_initializer=tf.keras.initializers.RandomUniform(-f2, f2)
        )(layer1_activation)

    batch2 = tf.keras.layers.BatchNormalization()(out)
    layer2_activation = tf.nn.relu(batch2)

    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    outputs = tf.keras.layers.Dense(
            1, activation="linear",
            kernel_initializer=last_init,
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )(layer2_activation)

    return tf.keras.Model([state_input, action_input], outputs)
