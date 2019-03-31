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

import cv2
#import helper file
import helper as hpr

# matplotlib inline
train_raw = pd.read_csv('train.csv')
test_raw = pd.read_csv('test.csv')

#print type(train_raw), type(test_raw)
print 'There are {}'.format(train_raw.shape[0]), \
       'samples for building the machine learning model and {}'.format(test_raw.shape[0]), \
       'samples for evaluating your model via Kaggle.'

# Let's look at the first 5 rows of train_raw dataset
train_raw.head(5)

# Preprocess the data to fit for the classifier
le = LabelEncoder().fit(train_raw.species) # Instantiate a LabelEncoder and fit to the given label data
labels = le.transform(train_raw.species)  # encode species strings and return labels with value between 0 and n_classes-1
classes = list(le.classes_)  # Save the species
test_ids = test_raw.id  # Save the image ids in test dataset

train = train_raw.drop(['id', 'species'], axis=1)
test = test_raw.drop(['id'], axis=1)


# Double check the data
print "The shapes of train and labels are: ", train.shape, labels.shape
print "There are {} species in total.".format(len(classes))
print "The shapes of test and test_ids are: ", test.shape, test_ids.shape

# construct the iterator
ss_split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
ss_split.get_n_splits(train, labels)

for train_index, test_index in ss_split.split(train, labels):
    X_train, X_test = train.values[train_index], train.values[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

# Double check the data
print y_train.shape, y_test.shape

# Test the leaf_image function
leaf_id = 343
leaf_img = hpr.leaf_image(leaf_id, target_length=160);
plt.imshow(leaf_img, cmap='gray'); plt.title('Leaf # '+str(leaf_id)); plt.axis('off'); plt.show()

# Test the function
species_img, label_info = hpr.species_image(0)  # Show this species of given index(0~98)
fig = plt.figure(num=None, figsize=(16, 3), dpi=1200, facecolor='w', edgecolor='w',frameon=False,linewidth = 0)
plt.imshow(species_img, cmap='gray'); plt.axis('off'); plt.show()
print label_info
#cv2.imwrite('species/'+label_info+'.jpg', species_img)  # Save the species image

species_img, label_info = hpr.species_image('Acer_Rubrum')  # show the species of give name
fig = plt.figure(num=None, figsize=(16, 3), dpi=1200, facecolor='w', edgecolor='w',frameon=False,linewidth = 0)
plt.imshow(species_img, cmap='gray'); plt.axis('off'); plt.show()

# List and save all the classes
for i, class_ in enumerate(classes):
    species_img, label_info = species_image(i)  # Show this species of given index(0~98)
    print label_info
    cv2.imwrite('species/'+label_info+'.jpg', species_img)

# Standardize the training data.
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

#param_grid = {'C':[1, 10],
#              'tol': [0.001, 0.0001]}
param_grid = {'C': [ 1000, 10000],
              'tol': [0.000001, 0.00001]}
log_reg = LogisticRegression(solver='newton-cg', multi_class='multinomial')
grid_search = GridSearchCV(log_reg, param_grid, scoring='neg_log_loss', refit='True', n_jobs=1, cv=ss_split)
grid_search.fit(X_train_scaled, y_train)

print 'Best parameter: {}'.format(grid_search.best_params_)
print 'Best cross-validation neg_log_loss score: {}'.format(grid_search.best_score_)
print '\nBest estimator:\n{}'.format(grid_search.best_estimator_)


scaler = StandardScaler().fit(X_test)
X_test_scaled = scaler.transform(X_test)

print 'ML Model: Logistic Regression'
# Accuracy
train_predictions = grid_search.predict(X_test_scaled)
acc = accuracy_score(y_test, train_predictions)
print 'Accuracy: {:.4%}'.format(acc)
# Logloss
train_predictions_p = grid_search.predict_proba(X_test_scaled)
ll = log_loss(y_test, train_predictions_p)
print 'Log Loss: {:.6}'.format(ll)

visualize_error(train_predictions, y_test)

# Standardize the training data.
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

# Using the optimal parameters
param_grid = {'C': [1000],
              'tol': [0.000001]}
log_reg = LogisticRegression(solver='newton-cg', multi_class='multinomial')
grid_search = GridSearchCV(log_reg, param_grid, scoring='neg_log_loss', refit='True', n_jobs=1, cv=ss_split)
grid_search.fit(X_train_scaled, y_train)

scaler = StandardScaler().fit(test)
test_scaled = scaler.transform(test)

test_predictions = grid_search.predict_proba(test_scaled)

# Format DataFrame
submission = pd.DataFrame(test_predictions, columns=classes)
submission.insert(0, 'id', test_ids)
submission.reset_index()

# Export Submission
submission.to_csv('submission_1208.csv', index = False)

# Double check the output
submission.head()
