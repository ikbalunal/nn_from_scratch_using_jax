#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename: train_nn_with_jax.py
# @Author: ikbal
# @Time: 6/11/2024 1:12 PM

import jax
import os
import jax.numpy as jnp
import tensorflow as tf
from jax import grad, pmap
from tqdm import tqdm

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=2'

class Config:
    def __init__(self):
        self.jax_seed = jax.random.PRNGKey(42)
        self.layer_units = [32, 16, 1]
        self.input_features = 13
        self.use_TPU = False
        self.batch_size = 202
        self.num_replicas = len(jax.devices())
        self.learning_rate = 0.05
        self.epochs = 30
        self.print_every = 5

class NeuralNetwork:
    def __init__(self, config):
        self.cfg = config
        self.weights_and_biases = self.initialize_w_b(config.layer_units)

    def initialize_w_b(self, layer_units):
        weights_and_biases = []
        seed = self.cfg.jax_seed
        for i, units in enumerate(layer_units):
            if i == 0:
                w = jax.random.uniform(key=seed, shape=(units, self.cfg.input_features), minval=-1.0, maxval=1.0, dtype=jnp.float32)
            else:
                w = jax.random.uniform(key=seed, shape=(units, layer_units[i-1]), minval=-1.0, maxval=1.0, dtype=jnp.float32)
            b = jax.random.uniform(key=seed, shape=(units,), minval=-1.0, maxval=1.0, dtype=jnp.float32)
            weights_and_biases.append([w, b])
        return weights_and_biases

    def linear(self, x, layer_weights_and_biases):
        w, b = layer_weights_and_biases
        return jnp.dot(x, w.T) + b

    def activation(self, x, layer_weights_and_biases, activation_fn=None):
        lin = self.linear(x, layer_weights_and_biases)
        if activation_fn is None:
            return lin
        elif activation_fn == 'relu':
            return jnp.maximum(jnp.zeros_like(a=lin), lin)
        else:
            return activation_fn(lin)

    def forward_pass(self, x, weights_and_biases):
        output_1 = self.activation(x, weights_and_biases[0], activation_fn='relu')
        output_2 = self.activation(output_1, weights_and_biases[1], activation_fn='relu')
        final_output = self.activation(output_2, weights_and_biases[2], activation_fn=None)
        return final_output

    def mse_loss(self, input, target, weights_and_biases):
        preds = self.forward_pass(input, weights_and_biases)
        return jnp.power(target - preds, 2).mean()

    def calculate_gradient(self, input, target):
        grads = grad(self.mse_loss)
        return grads(input, target, self.weights_and_biases)

    def update_weights(self, learning_rate, grads):
        for j in range(len(self.weights_and_biases)):
            self.weights_and_biases[j][0] = self.weights_and_biases[j][0] - learning_rate * grads[j][0]
            self.weights_and_biases[j][1] = self.weights_and_biases[j][1] - learning_rate * grads[j][1]

    def train_step(self, input, target, weights_and_biases, learning_rate):
        loss = self.mse_loss(input, target, weights_and_biases)
        grads = self.calculate_gradient(input, target)
        self.update_weights(learning_rate, grads)
        return loss

    def shard_data(self, x, y=None):
        features = x.shape[1]
        if y is not None:
            return x.reshape((self.cfg.num_replicas, self.cfg.batch_size, features)), y.reshape((self.cfg.num_replicas, self.cfg.batch_size, 1))
        return x.reshape((self.cfg.num_replicas, self.cfg.batch_size, features))

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(path="boston_housing.npz", test_split=0.2, seed=113)
    x_train = jnp.array(x_train, dtype=jnp.float32)
    y_train = jnp.array(y_train, dtype=jnp.float32)
    x_test = jnp.array(x_test, jnp.float32)
    y_test = jnp.array(y_test, jnp.float32)
    return x_train, y_train, x_test, y_test

def print_shapes(nn):
    for i, layer in enumerate(nn.weights_and_biases):
        print(f'Layer {i}  --->  Weight matrix: {layer[0].shape} || Bias vector : {layer[1].shape}')

if __name__ == '__main__':
    config = Config()
    print(f"JAX Version: {jax.__version__}")
    print(jax.devices())
    print(f"JAX seed : {config.jax_seed}")
    print(f"Layer units : {config.layer_units}")
    print(f"Input features : {config.input_features}")
    print(f"Use TPU : {config.use_TPU}")

    x_train, y_train, x_test, y_test = load_data()
    print(x_train.shape, y_train.shape)

    nn = NeuralNetwork(config)
    print_shapes(nn)

    # Test forward pass
    output = nn.activation(x_train[0], nn.weights_and_biases[0], activation_fn='relu')
    print(output)
    print(output.shape)
    output = nn.forward_pass(x_train[0], nn.weights_and_biases)
    print(output)
    print(output.shape)

    # Training loop
    print('Mean squared error on test set before training is : {}'.format(nn.mse_loss(x_test, y_test, nn.weights_and_biases)))

    for epoch in tqdm(range(config.epochs)):
        loss = nn.train_step(x_train, y_train, nn.weights_and_biases, config.learning_rate)
        if epoch % config.print_every == 0:
            print("MSE : {:.2f} at epoch {}".format(jnp.array(loss).mean(), epoch))

    print('Mean squared error on test set after training is : {}'.format(nn.mse_loss(x_test, y_test, nn.weights_and_biases)))

    # Parallel training
    nn.weights_and_biases = nn.initialize_w_b(config.layer_units)
    parallel_train_step = pmap(nn.train_step, in_axes=(0, 0, None))

    x_sharded, y_sharded = nn.shard_data(x_train, y_train)
    print(x_sharded.shape, y_sharded.shape)