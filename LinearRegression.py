import numpy as np
import math
import matplotlib.pyplot as plt


class Regression:
    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate


    def initialie_weights(self, n_features):
        self.theta1 = np.zeros((n_features, 1))
        self.theta0 = 0

    def fit(self, X, y):
        self.mse = []
        self.initialie_weights(X.shape[1])
        self.gradient_descent(self.n_iterations, X, y)


    def gradient_descent(self, n_iterations, X, y):
        for _ in range(n_iterations):
            y_pred = np.dot(X, self.theta1.T)
            mse = (1/(2*X.shape[0]))*np.sum(y_pred - y)**2
            self.mse.append(mse)
            grad_theta0 = (1/(X.shape[0]) * np.sum(y_pred - y))
            grad_theta1 = (1/(X.shape[0]) * np.sum((y_pred - y)*X))
            self.theta0 -= self.learning_rate * grad_theta0
            self.theta1 -= self.learning_rate * grad_theta1

    def predict(self, X):
        y_pred = np.dot(X, self.theta1) + self.theta0
        return y_pred

    def plot(self):
        plt.plot(self.mse)
        plt.title('Learning Curve')
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost')
        plt.show()


class LinearRegression(Regression):
    def __init__(self,  n_iterations=1000, learning_rate=0.0001):
        super(LinearRegression, self).__init__(n_iterations=n_iterations, learning_rate=learning_rate)

    def fit(self, X, y):
        super(LinearRegression, self).fit(X, y)

if __name__ == '__main__':
    X = np.array([1,2,3,4,5,6,9,8,7,4,5,9,6,5])
    X = X.reshape(X.shape[0],1)
    y = X*2
    regressor = LinearRegression()
    regressor.fit(X, y)
    y_pred = regressor.predict(50)
    print(regressor.mse)
    print(y_pred)
    regressor.plot()
