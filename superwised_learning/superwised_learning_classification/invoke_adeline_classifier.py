import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adeline_classifier import AdalineGD
from adeline_sgd_classifier import AdalineSGD

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

def model_with_adeline(X,y):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')

    ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Sum-squared-error')
    ax[1].set_title('Adaline - Learning rate 0.0001')
    plt.show()

def model_with_adeline_standardized_inputs(X,y):
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std() # Standardized inputs
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std() # Standardized inputs
    ada_gd = AdalineGD(n_iter=15, eta=0.01).fit(X_std, y)
    plot_decision_regions(X_std, y, classifier=ada_gd)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('sepallength[standardized]')
    plt.ylabel('petallength[standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.plot(range(1, len(ada_gd.cost_) + 1), ada_gd.cost_, marker ='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum - squared - error')
    plt.tight_layout()
    plt.show()



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
    # plt.show()

def model_with_adeline_stochastic_gradient_descent_inputs(X,y):
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std() # Standardized inputs
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std() # Standardized inputs
    ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    ada_sgd.fit(X_std, y)
    plot_decision_regions(X_std, y, classifier=ada_sgd)
    plt.title('Adaline - Stochastic Gradient Descent')
    plt.xlabel('sepallength[standardized]')
    plt.ylabel('petallength[standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.plot(range(1, len(ada_sgd.cost_) + 1), ada_sgd.cost_,marker ='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average Cost')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    iris_dataframe = read_iris_dataset()
    X, y = plot_iris_data(iris_dataframe)
    # ppn = model_with_adeline(X, y)
    # model_with_adeline_standardized_inputs(X,y)
    model_with_adeline_stochastic_gradient_descent_inputs(X,y)