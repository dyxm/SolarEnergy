# Created by Yuexiong Ding on 2018/7/30.
# 
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoLarsCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import scipy as sc

# columns = ['板温', '现场温度', '光照强度', '转换效率', '转换效率A', '转换效率B', '转换效率C', '电压A', '电压B', '电压C', '电流A', '电流B', '电流C', '功率A',
#            '功率B', '功率C', '平均功率', '风速', '风向', '发电量']
train_columns = ['板温', '光照强度', '电流A', '平均功率', '发电量']
# test_columns = ['ID', '板温', '现场温度', '光照强度', '转换效率', '转换效率A', '转换效率B', '转换效率C', '电压A', '电压B', '电压C', '电流A', '电流B',
#   '电流C', '功率A', '功率B', '功率C', '平均功率', '风速', '风向']
test_columns = ['ID', '板温', '光照强度', '电流A', '平均功率']
X_train = pd.read_csv('./DataSet/public.train.csv', usecols=train_columns)
test_data = pd.read_csv('./DataSet/public.test.csv', usecols=test_columns)
ID = test_data.pop('ID')
y_train = X_train.pop('发电量')
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

# X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.3)
# X_train = pd.DataFrame(X_train, columns=['板温', '光照强度',  '电流A', '电流B', '电流C', '功率A', '功率B', '功率C', '平均功率'])
# X_test = pd.DataFrame(X_test, columns=['板温', '光照强度',  '电流A', '电流B', '电流C', '功率A', '功率B', '功率C', '平均功率'])
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_train_1 = X_train[X_train['平均功率'] < 10000]
# X_test_1 = X_test[X_test['平均功率'] < 10000]
X_train_2 = X_train[X_train['平均功率'] >= 10000]
# X_test_2 = X_test[X_test['平均功率'] >= 10000]
y_train_1 = y_train.loc[~y_train.index.isin(X_train_2.index)]
# y_test_1 = y_test.loc[~y_test.index.isin(X_test_2.index)]
y_train_2 = y_train.loc[X_train_2.index]
# y_test_2 = y_test.loc[X_test_2.index]


# lasso_model = Lasso(alpha=0.000436321211122)
# lasso_model = LassoLarsCV()
# lasso_model.fit(X_train, y_train)
# print('系数矩阵:\n', lasso_model.coef_)
# print('线性回归模型:\n', lasso_model)
# print('最佳的alpha：', lasso_model.alpha_)
# y_pred = lasso_model.predict(X_test)
# linreg = LinearRegression()
# linreg.fit(X_train, y_train)
ridge_model_1 = RidgeCV()
ridge_model_2 = RidgeCV()
ridge_model_1.fit(X_train_1, y_train_1)
ridge_model_2.fit(X_train_2, y_train_2)
y_pred_1 = ridge_model_1.predict(X_train_1)
y_pred_2 = ridge_model_2.predict(X_train_2)
RMES_1 = mean_squared_error(y_train_1, y_pred_1)
RMES_2 = mean_squared_error(y_train_2, y_pred_2)
# plt.scatter(range(len(y_train_1)), y_train_1, c='y')
# plt.scatter(range(len(y_train_2)), y_train_2, c='y')
# ridge_model.fit(out, out_y)
# y_pred = ridge_model.predict(out)
# RMES = mean_squared_error(out_y, y_pred)
# plt.scatter(range(len(out_y)), out_y, c='y')

# for i in range(len(y_pred_2)):
#     if y_pred_2[i] > 15:
#         print(y_train_2[i])

# print(y_pred)
weight = len(y_train_2) / len(y_train)
print('RMES_1:', RMES_1)
print('\nRMES_2:', RMES_2)
print(1 / (1 + (RMES_1 * (1 - weight) + RMES_2 * weight) / 2))
# print(y_pred)
# print(y_test)
# plt.scatter(range(len(y_pred_1)), y_pred_1, c='r')
# plt.scatter(range(len(y_pred_2)), y_pred_2, c='r')
# plt.show()


# 预测test数据
# y_pred = ridge_model_1.predict(test_data)
# print(y_pred)
# plt.scatter(range(len(y_pred)), y_pred, c='r')
# plt.show()
y_pred = []
for i in range(len(test_data)):
    if test_data.loc[i]['平均功率'] >= 10000:
        y_pred_temp = ridge_model_2.predict(np.array(test_data.loc[i]).reshape(1, -1))
        # print(y_pred_temp)
    else:
        y_pred_temp = ridge_model_1.predict(np.array(test_data.loc[i]).reshape(1, -1))
    y_pred.append(y_pred_temp)

result = pd.DataFrame(ID, columns=['ID'])
result['Detection'] = np.array(y_pred).reshape(-1, 1)

result.to_csv('./predict3.csv', encoding='utf-8', index=False, header=False)
print(result)
