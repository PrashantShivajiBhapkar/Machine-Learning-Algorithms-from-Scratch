import numpy as np
import math
import pandas as pd


class LogisticRegression():
    def __init__(self, learning_rate=0.0001, epochs=100):
        self.alpha = learning_rate
        self.epochs = epochs

    def initialize_params(self, X):
        n_features = X.shape[1]
        self.W = np.random.uniform(-1/math.sqrt(n_features), 1/math.sqrt(n_features), (n_features, 1))

    def sigmoid(self, z):
        return (1/(1+np.exp(-z)))

    def fit(self, X, y):
        self.initialize_params(X)
        for _ in range(self.epochs):
            y_pred = self.sigmoid(np.dot(X, self.W))
            y = y.reshape(y_pred.shape)
            self.W -= (self.alpha * np.sum(np.dot((y_pred - y).T, X), axis=0, keepdims=True).T)

    def predict(self, X):
        y_pred = self.sigmoid(X.dot(self.W))
        return np.round(y_pred)

if __name__ == "__main__":
    df = pd.read_csv('C:/GitProjects/Machine-Learning-Algorithms-from-Scratch/Dataset/social_network_ads.csv')
    train_X = df.iloc[0:300, :-1].values
    train_y = df.iloc[0:300:, -1].values
    test_X = df.iloc[301:, :-1].values
    test_y = df.iloc[301:, -1].values
    regressor = LogisticRegression()
    regressor.fit(train_X, train_y)
    y_pred = regressor.predict(test_X)
    print(y_pred)
    print(regressor.W)
