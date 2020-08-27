import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

def plot(data, title):
    sns.set_style('dark')
    f, ax = plt.subplots()
    ax.set(ylabel='frequency')
    ax.set(xlabel='height(blue) / weight(green)')
    ax.set(title=title)
    sns.distplot(data[:, 0:1], color='blue')
    sns.distplot(data[:, 1:2], color='green')
    plt.savefig(title + '.png')
    plt.show()

np.random.seed(42)
height = np.random.normal(loc=168, scale=5, size=1000).reshape(-1, 1)
weight = np.random.normal(loc=70, scale=10, size=1000).reshape(-1, 1)

original_data = np.concatenate((height, weight), axis=1)
plot(original_data, 'Original')

standard_scaler_data = preprocessing.StandardScaler().fit_transform(original_data)
plot(standard_scaler_data, 'StandardScaler')

min_max_scaler_data = preprocessing.MinMaxScaler().fit_transform(original_data)
plot(min_max_scaler_data, 'MinMaxScaler')

max_abs_scaler_data = preprocessing.MaxAbsScaler().fit_transform(original_data)
plot(max_abs_scaler_data, 'MaxAbsScaler')

normalizer_data = preprocessing.Normalizer().fit_transform(original_data)
plot(normalizer_data, 'Normalizer')

robust_scaler_data = preprocessing.RobustScaler().fit_transform(original_data)
plot(robust_scaler_data, 'RobustScaler')