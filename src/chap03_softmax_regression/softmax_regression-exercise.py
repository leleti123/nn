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
y_p = np.random.normal(6., 1, dot_num)  #x和y坐标
y = np.ones(dot_num) #标签为1 
C1 = np.array([x_p, y_p, y]).T  # 组合成(x, y, label)格式

x_n = np.random.normal(6., 1, dot_num) # 从均值为6，标准差为1的高斯分布中采样x坐标，用于负样本
y_n = np.random.normal(3., 1, dot_num) # 从均值为3，标准差为1的高斯分布中采样y坐标，用于负样本
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


@tf.function
def train_one_step(model, optimizer, x_batch, y_batch):
    """
    一步梯度下降优化
    :param model: SoftmaxRegression 实例
    :param optimizer: 优化器（如 Adam, SGD）
    :param x_batch: 输入特征
    :param y_batch: 标签
    :return: 当前批次的损失与准确率
    """
    with tf.GradientTape() as tape:
        predictions = model(x_batch)
        loss, accuracy = compute_loss(predictions, y_batch)

    # 计算梯度并应用优化
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

# 生成网格点坐标矩阵
# x和y是1维数组，meshgrid将它们转换为2维网格坐标矩阵
# X和Y的形状都是(len(y), len(x))，其中X的每一行是x的复制，Y的每一列是y的复制
X, Y = np.meshgrid(x, y)

inp = np.array(list(zip(X.reshape(-1), Y.reshape(-1))), dtype=np.float32)
print(inp.shape)
Z = model(inp)
Z = np.argmax(Z, axis=1)
Z = Z.reshape(X.shape)
plt.contour(X,Y,Z)
plt.show()
