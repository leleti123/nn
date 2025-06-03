# python: 3.5.2
# encoding: utf-8

import numpy as np

def load_data(fname):

            x2 = float(line[1])

            # 将第三个字段转换为整数（标签/类别）
            t = int(line[2])


        # 将整个列表转换为 NumPy 数组，方便后续数值计算和处理
        return np.array(data)

def eval_acc(label, pred):

        pass
    

    def train(self, data_train):
        """
        训练 SVM 模型。
        :param data_train: 包含特征和标签的 NumPy 数组，形状为 (n_samples, n_features + 1)
        """

    def predict(self, x):
        """
        预测标签
        """
        # 请补全此处代码
        # 计算决策函数值


if __name__ == '__main__':
    # 载入数据，实际使用时将x替换为具体名称
    train_file = 'data/train_linear.txt'
    test_file = 'data/test_linear.txt'
    data_train = load_data(train_file)               # 数据格式[x1, x2, t]
    data_test = load_data(test_file)

    #print(data_train[:1000])  # 查看前5行数据

    # 使用训练集训练SVM模型
    svm = SVM()                                      # 初始化模型
    svm.train(data_train)                            # 训练模型

    # 使用SVM模型预测标签

    # 评估结果，计算准确率
    acc_train = eval_acc(t_train, t_train_pred)      # 计算训练集准确率
    acc_test = eval_acc(t_test, t_test_pred)         # 计算测试集准确率
    print("train accuracy: {:.1f}%".format(acc_train * 100))  # 打印结果
    print("test accuracy: {:.1f}%".format(acc_test * 100))
