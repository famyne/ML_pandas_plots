import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

from regression_models import LogisticRegression

bc = datasets.load_breast_cancer()

X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


regressor = LogisticRegression()
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)

print(f'LR classification accuracy: {accuracy(y_test, predicted)}')

