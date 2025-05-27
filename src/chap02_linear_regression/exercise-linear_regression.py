#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt


def load_data(filename): 
    """载入数据。"""
    xys = []
    with open(filename, 'r') as f:
        for line in f:
            line_data = list(map(float, line.strip().split()))
            xys.append(line_data)
    xs, ys = zip(*xys)
    return np.asarray(xs), np.asarray(ys)


def identity_basis(x):
    """恒等基函数"""
    ret = np.expand_dims(x, axis=1)
    return ret


def multinomial_basis(x, feature_num=10):
    """多项式基函数"""
    x = np.expand_dims(x, axis=1)
    feat = [x]
    for i in range(2, feature_num+1):
        feat.append(x**i)
    ret = np.concatenate(feat, axis=1)
    return ret


def gaussian_basis(x, feature_num=10):
    """高斯基函数"""
    centers = np.linspace(0, 25, feature_num)
    width = 1.0 * (centers[1] - centers[0])
    x = np.expand_dims(x, axis=1)
    x = np.concatenate([x]*feature_num, axis=1)
    
    out = (x-centers)/width
    ret = np.exp(-0.5 * out ** 2)
    return ret


def least_squares(phi, y, alpha=0.0):
    """最小二乘法优化权重w（含L2正则化）"""
    reg_matrix = alpha * np.eye(phi.shape[1])
    w = np.linalg.pinv(phi.T @ phi + reg_matrix) @ phi.T @ y
    return w


def gradient_descent(phi, y, lr=0.001, epochs=1000, alpha=0.0, tolerance=1e-6):
    """梯度下降优化权重w（含L2正则化和早停机制）"""
    w = np.zeros(phi.shape[1])
    n = len(y)
    
    # 特征归一化（关键改进：防止梯度爆炸）
    phi_mean = np.mean(phi, axis=0)
    phi_std = np.std(phi, axis=0)
    phi_std = np.where(phi_std == 0, 1, phi_std)  # 防止除以零
    phi_normalized = (phi - phi_mean) / phi_std
    
    prev_loss = float('inf')
    
    for epoch in range(epochs):
        y_pred = phi_normalized @ w
        error = y_pred - y
        
        # 计算梯度（含正则化项）
        gradient = (1/n) * phi_normalized.T @ error + alpha * w
        
        # 更新权重
        w -= lr * gradient
        
        # 计算当前损失
        current_loss = np.mean(error**2)
        
        # 打印阶段性损失
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {current_loss:.6f}")
        
        # 早停机制：如果损失变化很小，提前结束训练
        if abs(prev_loss - current_loss) < tolerance:
            print(f"Early stopping at epoch {epoch}")
            break
            
        prev_loss = current_loss
        
        # 防止梯度爆炸：如果损失变得无穷大，停止训练
        if np.isnan(current_loss) or np.isinf(current_loss):
            print(f"Gradient explosion at epoch {epoch}, reducing learning rate...")
            lr *= 0.1  # 降低学习率
            w = np.zeros(phi.shape[1])  # 重置权重
            epoch = 0  # 重新开始训练
    
    # 调整权重以适应原始特征尺度
    w = w / phi_std
    
    return w


def main(x_train, y_train, basis_type='identity', method='lsq', **kwargs):
    """主函数"""
    basis_func = {
        'identity': identity_basis,
        'polynomial': lambda x: multinomial_basis(x, **kwargs),
        'gaussian': lambda x: gaussian_basis(x, **kwargs)
    }[basis_type]
    
    # 特征预处理：添加偏置项
    phi = np.hstack([np.ones((len(x_train), 1)), basis_func(x_train)])
    
    # 选择优化方法
    if method == 'lsq':
        w = least_squares(phi, y_train, **kwargs)
    elif method == 'gd':
        w = gradient_descent(phi, y_train, **kwargs)
    else:
        raise ValueError("method必须为'lsq'或'gd'")
    
    # 定义预测函数
    def predict(x):
        phi_x = np.hstack([np.ones((len(x), 1)), basis_func(x)])
        return phi_x @ w
    
    return predict, w


def rmse(ys, ys_pred):
    """计算均方根误差 (RMSE)"""
    return np.sqrt(np.mean((ys - ys_pred) ** 2))


def mae(ys, ys_pred):
    """计算平均绝对误差 (MAE)"""
    return np.mean(np.abs(ys - ys_pred))


def r2(ys, ys_pred):
    """计算决定系数 (R²)"""
    ss_total = np.sum((ys - np.mean(ys)) ** 2)
    ss_residual = np.sum((ys - ys_pred) ** 2)
    return 1 - (ss_residual / ss_total)


if __name__ == '__main__':
    # 定义训练和测试数据文件路径
    train_file = 'train.txt'
    test_file = 'test.txt'
    
    # 载入数据
    x_train, y_train = load_data(train_file)
    x_test, y_test = load_data(test_file)
    print(f"训练集形状: {x_train.shape}")
    print(f"测试集形状: {x_test.shape}")

    # 训练模型（使用最小二乘法）
    f_lsq, w_lsq = main(x_train, y_train, method='lsq')
    
    # 训练模型（使用梯度下降法，降低学习率并增加早停）
    f_gd, w_gd = main(
        x_train, y_train, 
        method='gd', 
        lr=0.001,        # 降低学习率
        epochs=2000,     # 增加迭代次数
        alpha=0.01,      # 添加正则化
        tolerance=1e-6   # 设置早停阈值
    )
    
    # 评估最小二乘法
    y_train_pred_lsq = f_lsq(x_train)
    y_test_pred_lsq = f_lsq(x_test)
    
    # 评估梯度下降法
    y_train_pred_gd = f_gd(x_train)
    y_test_pred_gd = f_gd(x_test)
    
    # 打印评估结果
    print("\n=== 最小二乘法评估结果 ===")
    print(f"训练集 RMSE: {rmse(y_train, y_train_pred_lsq):.4f}")
    print(f"测试集 RMSE: {rmse(y_test, y_test_pred_lsq):.4f}")
    print(f"测试集 R²: {r2(y_test, y_test_pred_lsq):.4f}")
    
    print("\n=== 梯度下降法评估结果 ===")
    print(f"训练集 RMSE: {rmse(y_train, y_train_pred_gd):.4f}")
    print(f"测试集 RMSE: {rmse(y_test, y_test_pred_gd):.4f}")
    print(f"测试集 R²: {r2(y_test, y_test_pred_gd):.4f}")

    # 可视化结果
    plt.figure(figsize=(10, 6))
    
    # 训练数据点
    plt.scatter(x_train, y_train, c='red', s=30, alpha=0.5, label='训练数据')
    
    # 测试数据点
    plt.scatter(x_test, y_test, c='blue', s=30, alpha=0.5, label='测试数据')
    
    # 预测结果
    plt.plot(x_test, y_test_pred_lsq, 'k-', linewidth=2, label='最小二乘法预测')
    plt.plot(x_test, y_test_pred_gd, 'g--', linewidth=2, label='梯度下降预测')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('线性回归模型预测结果')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
