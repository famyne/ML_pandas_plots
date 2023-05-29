import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=200, n_features=1, noise=30, random_state=4)
X = 2*X
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=3234)

#fig = plt.figure(figsize=(8,6))
#plt.scatter(X[:,0],y, color = 'b', marker = 'o', s=30)
#plt.show()

print(X_train.shape)
print(y_train.shape)

from linear_regression import LinearRegression

regressor = LinearRegression(lr=0.01)
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)

def mse(y_true, y_predicted):
    return np.mean((y_true-y_predicted)**2)

mse_value = mse(y_test, predicted)
print(mse_value)

lr_list = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5]

quality = []
for lr in lr_list:
    a = LinearRegression(lr=lr)
    a.fit(X_train, y_train)
    pred = a.predict(X_test)
    #print(pred)
    mse_value = mse(y_test, pred)
    print(mse_value)
    quality.append(mse_value)
fig = plt.figure()
m1 = plt.scatter(lr_list, quality,s=10)
plt.xscale('log')


# y_pred_line = regressor.predict(X)
# fig = plt.figure()
# cmap = plt.get_cmap('viridis')
# m1 = plt.scatter(X_train, y_train, color=cmap(0.9),s=10)
# m2 = plt.scatter(X_test, y_test, color=cmap(0.5),s=10)
# plt.plot(X, y_pred_line, color='black', linewidth=2, label="Prediction")
plt.show()