import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoLarsCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import scipy as sc
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from scipy import optimize

train_columns = ['板温', '现场温度', '光照强度', '转换效率', '转换效率A', '转换效率B', '转换效率C', '电压A', '电压B', '电压C', '电流A', '电流B', '电流C',
                 '功率A',
                 '功率B', '功率C', '平均功率', '风速', '风向', '发电量']
# train_columns = ['板温', '光照强度', '电流A', '平均功率', '发电量']
test_columns = ['ID', '板温', '现场温度', '光照强度', '转换效率', '转换效率A', '转换效率B', '转换效率C', '电压A', '电压B', '电压C', '电流A', '电流B',
                '电流C', '功率A', '功率B', '功率C', '平均功率', '风速', '风向']
# test_columns = ['ID', '板温', '光照强度', '电流A', '平均功率']
X_train = pd.read_csv('./DataSet/public.train.csv', usecols=train_columns)
test_data = pd.read_csv('./DataSet/public.test.csv', usecols=test_columns)
ID = test_data.pop('ID')

# X_train = X_train[X_train['现场温度'] > -30]
# X_train = X_train[X_train['现场温度'] < 40]
# X_train = X_train[X_train['转换效率'] > 0]
# X_train = X_train[X_train['转换效率'] < 400]
# X_train = X_train[X_train['转换效率A'] < 400]
# X_train = X_train[X_train['转换效率B'] < 400]
# X_train = X_train[X_train['转换效率C'] < 400]
# X_train = X_train[X_train['平均功率'] < 10000]
# X_train = X_train[X_train['电压A'] > 580]
# X_train = X_train[X_train['电压A'] < 750]
# X_train = X_train[X_train['电压B'] > 580]
# X_train = X_train[X_train['电压B'] < 750]
# X_train = X_train[X_train['电压C'] > 580]
# X_train = X_train[X_train['电压C'] < 770]
# X_train = X_train[X_train['电流B'] < 10]
# X_train = X_train[X_train['电流C'] < 10]
# X_train = X_train[X_train['功率A'] < 6000]
# X_train = X_train[X_train['功率B'] < 6000]
# X_train = X_train[X_train['功率C'] < 6000]
# X_train = X_train[X_train['平均功率'] < 6000]


# test_data = test_data[test_data['现场温度'] > -30]
# test_data = test_data[test_data['现场温度'] < 50]
# test_data = test_data[test_data['转换效率'] < 400]
# test_data = test_data[test_data['转换效率A'] < 400]


y_train = X_train.pop('发电量')
print([x for x in X_train['转换效率']])


# print([1.0 / x for x in X_train['转换效率']])
# print(sc.stats.pearsonr([math.log(x) for x in X_train['转换效率']], [x for x in y_train]))


# plt.scatter(range(len(test_data['转换效率A'])), [x for x in test_data['转换效率A']], c='y')


def my_sigmoid(x, a, b, c):
    return a * (1.0 / (1 + np.exp(- (b * x) - c)))


x = np.array([x for x in X_train['板温']])
x1 = np.arange(-30, 40, 0.1)
y = np.array([x for x in y_train])
fita, fitb = optimize.curve_fit(my_sigmoid, x, y, (0, 0, 0))
plt.scatter(x, y, c='y')
plt.scatter(x, my_sigmoid(x, fita[0], fita[1], fita[2]), c='r')

# plt.plot(x1, my_sigmoid(x1, fita[0], fita[1], fita[2]))
plt.show()
# print(y)
# print(raw_data)

# plt.scatter(range(len(raw_data['板温'])), [x for x in raw_data['板温']], c='r')
# scaled_data = preprocessing.scale(raw_data)
# scaled_data = np.array(raw_data)
# for i in range(len(scaled_data[0])):
#     print(i)
#     print(sc.stats.pearsonr(scaled_data[:, i], y))

# pca = PCA(n_components=0.99)
# scaled_data = pca.fit_transform(scaled_data)
# print(len(scaled_data[0]))
# print(scaled_data)
