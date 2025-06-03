#!/usr/bin/env python
# coding: utf-8
import numpy as np
import tensorflow as tf

# ## 实现softmax函数（全流程使用float64）
def softmax(x: tf.Tensor) -> tf.Tensor:
    x = tf.cast(x, tf.float64)  # 统一为双精度
    max_per_row = tf.reduce_max(x, axis=-1, keepdims=True)
    shifted_logits = x - max_per_row  # 减法操作自动匹配类型
    exp_logits = tf.exp(shifted_logits)
    sum_exp = tf.reduce_sum(exp_logits, axis=-1, keepdims=True)
    return exp_logits / sum_exp

# 测试softmax
test_data = np.random.normal(size=[10, 5]).astype(np.float64)  # 初始数据即为float64
softmax_result = softmax(test_data)
tf_result = tf.nn.softmax(test_data, axis=-1)
print("Softmax验证:", np.allclose(softmax_result.numpy(), tf_result.numpy(), atol=1e-6))

# ## 实现sigmoid函数（全流程使用float64）
def sigmoid(x):
    x = tf.cast(x, tf.float64)
    return 1 / (1 + tf.exp(-x))

# 测试sigmoid
test_data = np.random.normal(size=[10, 5]).astype(np.float64)
sigmoid_result = sigmoid(test_data)
tf_result = tf.nn.sigmoid(test_data)
print("Sigmoid验证:", np.allclose(sigmoid_result.numpy(), tf_result.numpy(), atol=1e-6))

# ## 实现softmax交叉熵损失函数
def softmax_ce(logits, label):
    logits = tf.cast(logits, tf.float64)
    label = tf.cast(label, tf.float64)
    epsilon = 1e-12  # 双精度下使用更小的epsilon
    logits_max = tf.reduce_max(logits, axis=-1, keepdims=True)
    stable_logits = logits - logits_max
    exp_logits = tf.exp(stable_logits)
    prob = exp_logits / tf.reduce_sum(exp_logits, axis=-1, keepdims=True)
    loss = -tf.reduce_mean(tf.reduce_sum(label * tf.math.log(prob + epsilon), axis=1))
    return loss

# 测试softmax交叉熵
test_logits = np.random.normal(size=[10, 5]).astype(np.float64)
label = np.zeros_like(test_logits, dtype=np.float64)
label[np.arange(10), np.random.randint(0, 5, size=10)] = 1.0
tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=label, logits=test_logits  # 注意参数顺序：labels在前，logits在后
))
custom_loss = softmax_ce(test_logits, label)
print("Softmax交叉熵验证:", np.allclose(tf_loss.numpy(), custom_loss.numpy(), atol=1e-6))

# ## 实现sigmoid交叉熵损失函数
def sigmoid_ce(x, label):
    x = tf.cast(x, tf.float64)
    label = tf.cast(label, tf.float64)
    epsilon = 1e-12
    return -tf.reduce_mean(
        label * tf.math.log(x + epsilon) + (1 - label) * tf.math.log(1 - x + epsilon)
    )

# 测试sigmoid交叉熵
test_logits = np.random.normal(size=[10]).astype(np.float64)
label = np.random.randint(0, 2, 10).astype(np.float64)
prob = tf.nn.sigmoid(test_logits)
tf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=label, logits=test_logits  # 参数顺序：labels在前，logits在后
))
custom_loss = sigmoid_ce(prob, label)
print("Sigmoid交叉熵验证:", np.allclose(tf_loss.numpy(), custom_loss.numpy(), atol=1e-6))
