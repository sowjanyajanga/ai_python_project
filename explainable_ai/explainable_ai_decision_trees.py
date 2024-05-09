import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os

def train_test_split(X: pd.DataFrame ,y: pd.DataFrame, test_size, random_state):
    x_plus_y= pd.concat([X,y], axis=1)
    x_num_rows = len(x_plus_y.index)
    X_test = x_plus_y.sample(n=int(x_num_rows*test_size), random_state = 1)
    X_train = x_plus_y[~x_plus_y.index.isin(X_test.index)]
    y_test = X_test[['label']]
    y_train = X_train[['label']]
    X_test = X_test[['f1', 'f2', 'f3', 'f4']]
    X_train = X_train[['f1', 'f2', 'f3', 'f4']]
    print('X_train size:' + str(len(X_train.index)) + ' column count ' + str(len(X_train.columns)))
    print('X_test size ' + str(len(X_test.index)) + ' column count ' + str(len(X_test.columns)))
    print('y_train row count ' + str(len(y_train.index)) + ' column count ' + str(len(y_train.columns)))
    print('y_test row count' + str(len(y_test.index)) + ' column count ' + str(len(y_test.columns)))
    return X_train, X_test, y_train, y_test


if __name__=='__main__':
#     load home dataset
    col_names = ['f1', 'f2', 'f3', 'f4', 'label']

    pima = pd.read_csv('/Users/sowjanyaj/Documents/Study/AI_MachineLearning/AI_Python_WS/ai_python_project/explainable_ai/autopilot_data.txt',
                       header=None, names=col_names)

    for i in range(0, 100):
        xf1 = pima.at[i, 'f1']
        xf2 = pima.at[i, 'f2']
        xf3 = pima.at[i, 'f3']
        xf4 = pima.at[i, 'f4']
        xclass = pima.at[i, 'label']

    b1 = 1.5; b2 = 1.5; b3 = 0.1; b4 = 0.1

    xf1 = round(xf1 * b1, 2)
    xf2 = round(xf2 * b2, 2)
    xf3 = round(xf3 * b3, 2)
    xf4 = round(xf4 * b4, 2)

    # print(pima.head())
    feature_cols = ['f1', 'f2', 'f3', 'f4']
    X = pima[feature_cols] # Features
    y = pima.label # Target variable
    # print(X)
    # print(y)
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.3, random_state=1) # 70% training and 30% test

    # Create a default decision tree classifier with no hyper-parameters
    # This classifier is going to learn on the training data set to come up with a decision tree structure
    # using gini impurities.
    # But in actuality you would want to tune it to a max depth and number of splits inorder to have a
    # workable decision tree to work with
#     estimator = DecisionTreeClassifier(max_depth=2, max_leaf_nodes=3,
# min_samples_leaf=100)
    estimator = DecisionTreeClassifier()
    print(estimator)
    estimator.fit(X_train, y_train)
    print('predict')
    y_pred = estimator.predict(X_test)

    X_DL = [[xf1, xf2, xf3, xf4]]
    prediction = estimator.predict(X_DL)
    e = False
    print(y_pred)

    if (prediction == xclass):
        e = True
        t += 1
    if (prediction != xclass):
        e = False
        f += 1

    print('accuracy ', metrics.accuracy_score(y_pred, y_test))

    from matplotlib.pyplot import figure
    plt.figure(dpi=400, edgecolor="r", figsize=(10, 10))
    F = ["f1", "f2", "f3", "f4"]
    C = ["Right", "Left"]
    # plot_tree(estimator, filled=True, feature_names=F, rounded=True,
    # precision=2, fontsize=3, proportion=True, max_depth=None,
    # class_names=C)
    # plt.savefig('dt.png')
    # plt.show()
    plot_tree(estimator, filled=True, feature_names=F, rounded=True,
    precision=2, fontsize=3, proportion=True, max_depth=2,
    class_names=C)
    plt.savefig('dt.png')
    plt.figure(dpi=400, edgecolor="r", figsize=(3, 3))
    plt.show()
