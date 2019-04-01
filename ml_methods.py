# Importing some useful/necessary packages
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV

from sklearn.metrics import accuracy_score, log_loss

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import helper as hpr

def run_naive_bayes(train, test, ss_split, labels):
    # prepare training and test data
    X_train, X_test, y_train, y_test = hpr.prepData(train, test, ss_split, labels);

    clf = GaussianNB().fit(X_train, y_train) # Instantiate a classifier and fit this classifier to the data
    print ('ML Model: Naive Bayes')
    # Cross-validation
    scores = cross_val_score(GaussianNB(), train.values, labels, cv=ss_split)
    print ('Mean Cross-validation scores: {}'.format(np.mean(scores)))
    # Accuracy
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    # Logloss
    train_predictions_p = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions_p)

    test_predictions = clf.predict_proba(test)
    return test_predictions, acc, ll

def run_support_vector_machine(train, test, ss_split, labels):
    # prepare training and test data
    X_train, X_test, y_train, y_test = hpr.prepData(train, test, ss_split, labels);
    clf = SVC(probability=True)

    # Gird search
    #param_grid = {'C': [1, 10, 100, 1000, 10000, 100000],
    #              'gamma': [1, 10, 100, 1000, 10000, 100000]}
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

    grid_search = GridSearchCV(SVC(probability=True), param_grid=param_grid, cv=ss_split)
    grid_search.fit(X_train, y_train)

    print ('Best parameter: {}'.format(grid_search.best_params_))
    print ('Best cross-validation accuracy score: {}'.format(grid_search.best_score_))
    print ('\nBest estimator:\n{}'.format(grid_search.best_estimator_))
    # results = pd.DataFrame(grid_search.cv_results_)
    # Show the first 5 rows of the result
    #print results.head()

    # scores = np.array(results.mean_test_score).reshape(6, 6)
    #
    # ax = sns.heatmap(scores, annot=True, fmt=".2f",linewidths=.5);
    # ax.invert_yaxis()
    # ax.set(xticklabels=param_grid['gamma']); ax.set(yticklabels=param_grid['C'])
    # plt.yticks(rotation=0)
    # plt.xlabel('gamma'); plt.ylabel('C'); plt.show()

    print ('ML Model: Suppoort Vector Machine')
    # Accuracy
    train_predictions = grid_search.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    # Logloss
    train_predictions_p = grid_search.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions_p)

    test_predictions = grid_search.predict_proba(test)
    return test_predictions, acc, ll

def run_logistic_regression(train, test, ss_split, labels):
    # prepare training and test data
    X_train, X_test, y_train, y_test = hpr.prepData(train, test, ss_split, labels);

    #param_grid = {'C':[1, 10],
    #              'tol': [0.001, 0.0001]}

    # Standardize the training data.
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    scaler = StandardScaler().fit(X_test)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {'C': [ 1000, 10000],
                  'tol': [0.000001, 0.00001]}
    log_reg = LogisticRegression(solver='newton-cg', multi_class='multinomial')
    grid_search = GridSearchCV(log_reg, param_grid, scoring='neg_log_loss', refit='True', n_jobs=1, cv=ss_split)
    grid_search.fit(X_train_scaled, y_train)

    print ('Best parameter: {}'.format(grid_search.best_params_))
    print ('Best cross-validation neg_log_loss score: {}'.format(grid_search.best_score_))
    print ('\nBest estimator:\n{}'.format(grid_search.best_estimator_))
    print ('ML Model: Logistic Regression')
    # Accuracy
    train_predictions = grid_search.predict(X_test_scaled)
    acc = accuracy_score(y_test, train_predictions)
    # Logloss
    train_predictions_p = grid_search.predict_proba(X_test_scaled)
    ll = log_loss(y_test, train_predictions_p)

    scaler = StandardScaler().fit(test)
    test_scaled = scaler.transform(test)
    test_predictions = grid_search.predict_proba(test_scaled)

    # visualize error
    # hpr.visualize_error(train_predictions, y_test)

    return test_predictions, acc, ll

def run_k_nearest_neighbours(train, test, ss_split, labels):
    # prepare training and test data
    X_train, X_test, y_train, y_test = hpr.prepData(train, test, ss_split, labels);

    clf = KNeighborsClassifier(3)  # Instantiate a classifier
    clf.fit(X_train, y_train) # Fit this classifier to the data
    print ('ML Model: K-Nearest Neighbours')

    # Cross-validation
    scores = cross_val_score(KNeighborsClassifier(3), train.values, labels, cv=ss_split)
    #print 'Mean Cross-validation scores: {}'.format(np.mean(scores))

    # Accuracy
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    # Logloss
    train_predictions_p = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions_p)

    test_predictions = clf.predict_proba(test)
    return test_predictions, acc, ll

def run_linear_discriminant_analysis(train, test, ss_split, labels):
    # prepare training and test data
    X_train, X_test, y_train, y_test = hpr.prepData(train, test, ss_split, labels);

    clf = LinearDiscriminantAnalysis().fit(X_train, y_train)
    print ('ML Model: Linear Discriminant Analysis')

    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)

    train_predictions_p = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions_p)

    test_predictions = clf.predict_proba(test)
    return test_predictions, acc, ll
