import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 生成一些随机数据（例如，模拟生成的样本）
np.random.seed(0)
generated_samples = np.random.exponential(scale=2, size=1000)  # 假设这是非正态分布的数据


# Box-Cox变换函数
def box_cox_transform(samples):
    # 计算最佳lambda值
    lambda_optimal = stats.boxcox_normmax(samples)  #该函数通过不同的统计方法（如最大似然估计、Pearson相关、标准化的正态分布等）来寻找最佳的 λ 值，以使得 Box-Cox 变换后的数据更接近正态分布。

    # 进行Box-Cox变换
    transformed_samples = stats.boxcox(samples, lmbda=lambda_optimal)
    return transformed_samples, lambda_optimal


# 逆变换函数
def inverse_box_cox(transformed_samples, lambda_optimal):
    if lambda_optimal == 0:
        return np.exp(transformed_samples)
    else:
        return (transformed_samples * lambda_optimal + 1) ** (1 / lambda_optimal)


# 应用Box-Cox变换
transformed_samples, lambda_optimal = box_cox_transform(generated_samples)


# 绘制直方图和Q-Q图
def plot_results(original_samples, transformed_samples, lambda_optimal):
    plt.figure(figsize=(12, 6))

    # 原始样本直方图
    plt.subplot(1, 2, 1)
    sns.histplot(original_samples, bins=30, kde=True, color='blue', stat="density")
    plt.title('Original Samples Histogram')
    plt.xlabel('Value')
    plt.ylabel('Density')

    # 变换后样本直方图
    plt.subplot(1, 2, 2)
    sns.histplot(transformed_samples, bins=30, kde=True, color='orange', stat="density")
    plt.title('Transformed Samples Histogram (Box-Cox)')
    plt.xlabel('Value')
    plt.ylabel('Density')

    plt.tight_layout()
    plt.show()

    # Q-Q图
    plt.figure(figsize=(6, 6))
    stats.probplot(transformed_samples, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Transformed Samples')
    plt.show()


# 打印结果
print("Optimal Lambda:", lambda_optimal)
print("Transformed Samples (first 5):", transformed_samples[:5])

# 绘制结果
plot_results(generated_samples, transformed_samples, lambda_optimal)
