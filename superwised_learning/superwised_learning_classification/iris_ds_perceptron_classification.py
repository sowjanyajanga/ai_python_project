import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron_classifier import Perceptron

def read_iris_dataset():
    s = os.path.join('https://archive.ics.uci.edu', 'ml', 'machine-learning-databases','iris', 'iris.data')
    print('URL:', s)
    df = pd.read_csv(s, header=None,encoding='utf-8')
    df.tail()
    return df

def plot_iris_data(df):
    # select setosa and versicolor
    y = df.iloc[0:100, 4].values # returns 5th column for 0-100 rows
    y = np.where(y == 'Iris-setosa', -1, 1)
    # extract sepal length(1st column) and petal length(third column)
    X = df.iloc[0:100, [0, 2]].values # 1st column and third column of first 100 rows
    # plot data
    # plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    # plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    # plt.xlabel('sepal length[cm]')
    # plt.ylabel('petal length[cm]')
    # plt.legend(loc='upper left')
    # plt.show()
    return X,y # X[0] petal length, X[1] sepal length, y = known actual flower type

def model_with_perceptor(X,y):
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker ='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number  of  updates')
    # plt.show()
    return ppn

def plot_decision_regions(X, y, classifier, resolution=0.02):
    from matplotlib.colors import ListedColormap
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8,c=colors[idx],marker=markers[idx],label=cl,edgecolor='black')
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    iris_dataframe = read_iris_dataset()
    X, y = plot_iris_data(iris_dataframe)
    ppn = model_with_perceptor(X, y)
    plot_decision_regions(X,y,ppn)
