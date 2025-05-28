# python: 3.5.2
# encoding: utf-8

import numpy as np


def load_data(fname):
    """
    载入并处理数据
    """
    with open(fname, 'r') as f:           # 以可读方式访问文件
        data = []                         # 创建空白列表，为储存数据做准备
        line = f.readline()               # 首行是标题行，自动跳过
        for line in f:                    # 从第二行开始对每一行进行循环
            line = line.strip().split()   # 去除行首行尾的空白字符，将处理后的行按空白字符分割成字符串列表
   #从分割后的列表中提取三个值：前两个值转换为浮点数，作为特征x1和x2。第三个值转换为整数，作为标签t。
            x1 = float(line[0])   
            x2 = float(line[1])
            t = int(line[2])
            data.append([x1, x2, t])     # 将解析后的特征和标签组合成一个列表，添加到data列表中
        return np.array(data)            # 将收集的所有数据行转换为 NumPy 数组并返回


def eval_acc(label, pred):
    """
    计算准确率，方便评估分类模型的性能
    """
    return np.sum(label == pred) / len(pred) #准确率 = 正确预测的样本数 / 总样本数，#label == pred是比较预测值与真实标签是否相等，返回布尔数组


class SVM():
    """
    SVM模型。
    """
    #目标函数：(1/2)||w||² + C * Σmax(0, 1 - y_i(w·x_i + b))，这是 SVM 的优化目标，包含两部分：防止过拟合和惩罚分类错误
    def __init__(self):
        # 请补全此处代码
        self.w = None                  # 初始化权重向量为None，后续训练时会设置
        self.b = 0                     # 偏置项
        pass

    def train(self, data_train):
        """
        训练模型。
        """
        # 提取特征和标签
        x = data_train[:, :2]           # 从训练数据中提取特征（前两列）
        t = data_train[:, 2]            # 从训练数据中提取标签（第三列）
        
        # 初始化参数
        n_samples, n_features = x.shape
        self.w = np.zeros(n_features)   # 将权重向量初始化为零向量
        learning_rate = 0.01            # 学习率η：控制参数更新步长
        lambda_ = 0.01                  # 正则化系数λ：控制模型复杂度         
        epochs = 1000                   # 训练轮数，即遍历整个数据集的次数
 
        # 梯度下降优化
        for _ in range(epochs):                                                 #外层循环 epochs 次，控制训练轮数
            for idx, x_i in enumerate(x):                                       #内层遍历每个样本，进行参数更新
                # 计算预测值（使用符号函数判断分类）                               # idx 为第几个真实标签，表示是否分类正确了，np.dot(x_i, self.w) + self.b为模型预测值，分类正确，condition大于等于1
                condition = t[idx] * (np.dot(x_i, self.w) + self.b)
                if condition >= 1:
                    self.w -= learning_rate * (lambda_ * self.w)                # 正确的话仅更新正则化项
                else:
                    self.w -= learning_rate * (lambda_ * self.w - t[idx] * x_i) # 更新正则化项，惩罚权重过大，惩罚分类错误
                    self.b -= learning_rate * (-t[idx])                         # 更新损失项
        # 请补全此处代码

        def predict(self, x):

            if self.weights is None:
                raise RuntimeError("Model not trained yet")                     #确保模型已训练（权重已初始化）
            logits = np.dot(x, self.weights) + self.bias                        #logits 即线性函数 w*x + b，logits > 0，预测为正类，反之负类
            return (logits >= 0).astype(int)                                    #将布尔值（True/False）转换为整数（1/0）
        # 二分类示例
    def predict(self, x):
        """
        预测标签。
        """

        # 请补全此处代码
        # 计算决策函数值
        decision_values = np.dot(x, self.w) + self.b                            # 计算决策函数值:logits = X·w + b,其中 w 是权重向量,b 是偏置项
        return np.where(decision_values >= 0, 1, 0)                             # 如果大于等于 0，返回 1（正类）；否则返回 0（负类）


if __name__ == '__main__':
    # 载入数据，实际使用时将x替换为具体名称
    train_file = 'data/train_linear.txt'
    test_file = 'data/test_linear.txt'
    data_train = load_data(train_file)               # 数据格式[x1, x2, t]
    data_test = load_data(test_file)

    # 使用训练集训练SVM模型
    svm = SVM()                                      # 初始化模型
    svm.train(data_train)                            # 训练模型

    # 使用SVM模型预测标签
    x_train = data_train[:, :2]                      # 提取训练集特征feature [x1, x2]
    t_train = data_train[:, 2]                       # 提取训练集真实标签
    t_train_pred = svm.predict(x_train)              # 预测训练集标签
    x_test = data_test[:, :2]                        # 提取测试集特征
    t_test = data_test[:, 2]                         # 提取测试集真实标签
    t_test_pred = svm.predict(x_test)                # 预测测试集集标签

    # 评估结果，计算准确率
    acc_train = eval_acc(t_train, t_train_pred)      # 计算训练集准确率
    acc_test = eval_acc(t_test, t_test_pred)         # 计算测试集准确率
    print("train accuracy: {:.1f}%".format(acc_train * 100))  # 打印结果
    print("test accuracy: {:.1f}%".format(acc_test * 100))
