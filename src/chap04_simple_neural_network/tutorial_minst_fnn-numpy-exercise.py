#!/usr/bin/env python
# coding: utf-8
# ## 准备数据
# In[1]:

import os
import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
#定义了一个函数mnist_dataset()，用于加载并预处理MNIST数据集
def mnist_dataset():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    #normalize
    x = x/255.0
    x_test = x_test/255.0
    
    return (x, y), (x_test, y_test)

# ## Demo numpy based auto differentiation
# In[3]:
import numpy as np

# 定义矩阵乘法层
class Matmul:
    def __init__(self):
        self.mem = {}
        
    def forward(self, x, W):
        # 前向传播：执行矩阵乘法，计算 h = x @ W
        h = np.matmul(x, W)
        # 缓存输入 x 和 权重 W，以便在反向传播中计算梯度
        self.mem={'x': x, 'W':W}
        # 缓存输入 x 和 权重 W，以便在反向传播中计算梯度
        return h
    
    def backward(self, grad_y):
        '''
        x: shape(N, d)
        w: shape(d, d')
        grad_y: shape(N, d')
        '''
       # 反向传播计算 x 和 W 的梯度
        x = self.mem['x']
        W = self.mem['W']
        
        '''计算矩阵乘法的对应的梯度'''
        grad_x = np.matmul(grad_y, W.T)
        grad_W = np.matmul(x.T, grad_y)
      
        return grad_x, grad_W

# 定义 ReLU 激活层
class Relu:
    def __init__(self):
        self.mem = {}
        
    def forward(self, x):
        self.mem['x']=x
        return np.where(x > 0, x, np.zeros_like(x))
    
    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        ####################
        '''计算relu 激活函数对应的梯度'''
        x = self.mem['x']
        grad_x = grad_y * (x > 0)  # ReLU的梯度是1（x>0）或0（x<=0）
        ####################
        return grad_x

# 定义 Softmax 层（输出概率）
class Softmax:
    '''
    softmax over last dimention
    '''
    def __init__(self):
        self.epsilon = 1e-12
        self.mem = {}
        
    def forward(self, x):
        '''
        x: shape(N, c)
        '''
        x_exp = np.exp(x)
        partition = np.sum(x_exp, axis=1, keepdims=True)
        out = x_exp/(partition+self.epsilon)
        
        self.mem['out'] = out
        self.mem['x_exp'] = x_exp
        return out
    
    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        s = self.mem['out']
        sisj = np.matmul(np.expand_dims(s,axis=2), np.expand_dims(s, axis=1)) # (N, c, c)
        # 对grad_y进行维度扩展
        # 假设grad_y是一个形状为(N, c)的梯度张量
        # np.expand_dims(grad_y, axis=1)将其形状变为(N, 1, c)
        g_y_exp = np.expand_dims(grad_y, axis=1)
        tmp = np.matmul(g_y_exp, sisj) #(N, 1, c)
        tmp = np.squeeze(tmp, axis=1)
        tmp = -tmp + grad_y * s 
        return tmp
    
# 定义 Log 层（计算 log softmax，用于交叉熵）
class Log:
    '''
    softmax over last dimention
    '''
    def __init__(self):
        self.epsilon = 1e-12
        self.mem = {}
        
    def forward(self, x):
        '''
        x: shape(N, c)
        '''
        out = np.log(x+self.epsilon)
        
        self.mem['x'] = x
        return out
    
    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        x = self.mem['x']
        
        return 1./(x+1e-12) * grad_y

import tensorflow as tf

# 先定义 x，再创建 label
x = np.random.normal(size=[5, 6])  # 定义 x
label = np.zeros_like(x)
label[0, 1] = 1.
label[1, 0] = 1
label[2, 3] = 1
label[3, 5] = 1
label[4, 0] = 1

W1 = np.random.normal(size=[6, 5])
W2 = np.random.normal(size=[5, 6])

mul_h1 = Matmul()
mul_h2 = Matmul()
relu = Relu()
softmax = Softmax()
log = Log()

h1 = mul_h1.forward(x, W1)  # shape(5, 4)
h1_relu = relu.forward(h1)
h2 = mul_h2.forward(h1_relu, W2)
h2_soft = softmax.forward(h2)
h2_log = log.forward(h2_soft)

h2_log_grad = log.backward(label)
h2_soft_grad = softmax.backward(h2_log_grad)
h2_grad, W2_grad = mul_h2.backward(h2_soft_grad)
h1_relu_grad = relu.backward(h2_grad)
h1_grad, W1_grad = mul_h1.backward(h1_relu_grad)

print(h2_log_grad)
print('--' * 20)
# print(W2_grad)

with tf.GradientTape() as tape:
    x_tf, W1_tf, W2_tf, label_tf = tf.constant(x), tf.constant(W1), tf.constant(W2), tf.constant(label)
    tape.watch(W1_tf)
    tape.watch(W2_tf)
    h1_tf = tf.matmul(x_tf, W1_tf)
    h1_relu_tf = tf.nn.relu(h1_tf)
    h2_tf = tf.matmul(h1_relu_tf, W2_tf)
    prob_tf = tf.nn.softmax(h2_tf)
    log_prob_tf = tf.math.log(prob_tf)
    loss_tf = tf.reduce_sum(label_tf * log_prob_tf)
    grads = tape.gradient(loss_tf, [W1_tf, W2_tf])
    print("TensorFlow W1 Gradient:", grads[0].numpy())
    print("TensorFlow W2 Gradient:", grads[1].numpy())
