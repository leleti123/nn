#!/usr/bin/env python
# coding: utf-8

# # Softmax Regression Example

# ### 生成数据集， 看明白即可无需填写代码
# #### '<font color="blue">+</font>' 从高斯分布采样 (X, Y) ~ N(3, 6, 1, 1, 0).<br>
# #### '<font color="green">o</font>' 从高斯分布采样  (X, Y) ~ N(6, 3, 1, 1, 0)<br>
# #### '<font color="red">*</font>' 从高斯分布采样  (X, Y) ~ N(7, 7, 1, 1, 0)<br>

import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
import matplotlib.cm as cm
import numpy as np
# get_ipython().run_line_magic('matplotlib', 'inline')

# ##数据处理
dot_num = 100 # 设置数据点数量
x_p = np.random.normal(3., 1, dot_num) # 从均值为3，标准差为1的高斯分布中采样x坐标，用于正样本
y_p = np.random.normal(6., 1, dot_num)
y = np.ones(dot_num)
C1 = np.array([x_p, y_p, y]).T

x_n = np.random.normal(6., 1, dot_num) # 从均值为6，标准差为1的高斯分布中采样x坐标，用于负样本
y_n = np.random.normal(3., 1, dot_num)
y = np.zeros(dot_num)
C2 = np.array([x_n, y_n, y]).T

x_b = np.random.normal(7., 1, dot_num) # 从均值为7，标准差为1的高斯分布中采样x坐标，用于负样本
y_b = np.random.normal(7., 1, dot_num)
y = np.ones(dot_num)*2
C3 = np.array([x_b, y_b, y]).T

plt.scatter(C1[:, 0], C1[:, 1], c='b', marker='+') # 绘制正样本，用蓝色加号表示
plt.scatter(C2[:, 0], C2[:, 1], c='g', marker='o') # 绘制负样本，用绿色圆圈表示（将c='p'改为c='g'）
plt.scatter(C3[:, 0], C3[:, 1], c='r', marker='*') # 绘制负样本，用红色星号表示

data_set = np.concatenate((C1, C2, C3), axis=0) # 将正样本和负样本连接成一个数据集
np.random.shuffle(data_set) # 随机打乱数据集的顺序


# ## 建立模型
# 建立模型类，定义loss函数，定义一步梯度下降过程函数
# 
# 填空一：在`__init__`构造函数中建立模型所需的参数
# 
# 填空二：实现softmax的交叉熵损失函数(不使用tf内置的loss 函数)

epsilon = 1e-12 # 定义一个极小值，用于后续防止数值计算中出现如对数运算时输入为 0 的情况，保证数值稳定性
class SoftmaxRegression():
    def __init__(self):
        '''============================='''
        # 硬编码输入特征数为2（数据集是2维坐标），输出类别数为3（标签0、1、2）
        input_dim = 2  # 输入特征数（x和y坐标）
        output_dim = 3 # 输出类别数（3类样本）
        
        # 使用动态维度计算 Xavier 初始化范围
        limit = tf.sqrt(6.0 / (input_dim + output_dim))
        
        # 初始化权重 W：形状[input_dim, output_dim]
        self.W = tf.Variable(
            initial_value=tf.random.uniform(
                shape=[input_dim, output_dim],
                minval=-limit,
                maxval=limit
            ),
            regularizer=tf.keras.regularizers.l2(0.01),
            name='weights'
        )
        
        # 初始化偏置 b：形状[output_dim]
        self.b = tf.Variable(
            initial_value=tf.random.uniform(
                shape=[output_dim],
                minval=-0.01,
                maxval=0.01
            ),
            name='bias'
        )
        '''============================='''
        
        self.trainable_variables = [self.W, self.b]
    
    @tf.function # 这是一个装饰器，作用是将该函数编译为 TensorFlow 的静态图模式，这样可以提高函数的执行效率，减少运行时的开销
    def __call__(self, inp):
        logits = tf.matmul(inp, self.W) + self.b # shape(N, 3)
        pred = tf.nn.softmax(logits)
        return pred    
    
@tf.function
def compute_loss(pred, label):
    label = tf.one_hot(tf.cast(label, dtype=tf.int32), dtype=tf.float32, depth=3)
    '''============================='''
    # 实现softmax交叉熵损失（直接使用pred计算，避免重复计算logits）
    pred = tf.clip_by_value(pred, epsilon, 1.0)  # 防止log(0)
    losses = -tf.reduce_sum(label * tf.math.log(pred), axis=1)  # 交叉熵公式
    '''============================='''
    # 计算平均损失
    loss = tf.reduce_mean(losses) 
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(label,axis=1), tf.argmax(pred, axis=1)), dtype=tf.float32))
    return loss, accuracy

@tf.function
def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        pred = model(x)
        loss, accuracy = compute_loss(pred, y)
        
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, accuracy


# ### 实例化一个模型，进行训练

model = SoftmaxRegression()  # 无需传入参数，硬编码在__init__中
opt = tf.keras.optimizers.SGD(learning_rate=0.01)
x1, x2, y = list(zip(*data_set))
x = np.array(list(zip(x1, x2)), dtype=np.float32)  # 转换为 float32
y = np.array(y, dtype=np.int32)  # 转换为 int32

for i in range(1000):
    loss, accuracy = train_one_step(model, opt, x, y)
    if i%50==49:
        print(f'loss: {loss.numpy():.4f}\t accuracy: {accuracy.numpy():.4f}')

# ## 结果展示，无需填写代码

plt.scatter(C1[:, 0], C1[:, 1], c='b', marker='+')
plt.scatter(C2[:, 0], C2[:, 1], c='g', marker='o')
plt.scatter(C3[:, 0], C3[:, 1], c='r', marker='*')

x = np.arange(0., 10., 0.1)
y = np.arange(0., 10., 0.1)

X, Y = np.meshgrid(x, y)
inp = np.array(list(zip(X.reshape(-1), Y.reshape(-1))), dtype=np.float32)
print(inp.shape)
Z = model(inp)
Z = np.argmax(Z, axis=1)
Z = Z.reshape(X.shape)
plt.contour(X,Y,Z)
plt.show()
