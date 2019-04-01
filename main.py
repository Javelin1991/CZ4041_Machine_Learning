# Importing some useful/necessary packages
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, log_loss

import cv2
import helper as hpr
import ml_methods as ML_Methods

print("\nPlease select one machine learning algorithm below to perform leaf classification.\n")
print("NB\t:\tNaive Bayes\n")
print("SVM\t:\tSupport Vector machine\n")
print("LR\t:\tLogistic Regression\n")
print("KNN\t:\tK-nearest Neighbours\n")
print("LDA\t:\tLinear Discriminant Analysis\n\n")
print("Example: if you would like to run Naive Bayes, please type NB\n")

algo = input()

# matplotlib inline
train_raw = pd.read_csv('train.csv')
test_raw = pd.read_csv('test.csv')

#print type(train_raw), type(test_raw)
print ('There are {}'.format(train_raw.shape[0]), \
       'samples for building the machine learning model and {}'.format(test_raw.shape[0]), \
       'samples for evaluating your model via Kaggle.')

# Let's look at the first 5 rows of train_raw dataset
train_raw.head(5)

###################### Preprocessing Data ######################
le = LabelEncoder().fit(train_raw.species) # Instantiate a LabelEncoder and fit to the given label data
labels = le.transform(train_raw.species)  # encode species strings and return labels with value between 0 and n_classes-1
classes = list(le.classes_)  # Save the species
test_ids = test_raw.id  # Save the image ids in test dataset

train = train_raw.drop(['id', 'species'], axis=1)
test = test_raw.drop(['id'], axis=1)


# Double check the data
print ("The shapes of train and labels are: ", train.shape, labels.shape)
print ("There are {} species in total.".format(len(classes)))
print ("The shapes of test and test_ids are: ", test.shape, test_ids.shape)

# construct the iterator
ss_split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
ss_split.get_n_splits(train, labels)


# List and save all the classes
for i, class_ in enumerate(classes):
    species_img, label_info = hpr.species_image(i, labels, train_raw, classes)  # Show this species of given index(0~98)
    print (label_info)
    cv2.imwrite('species/'+label_info+'.jpg', species_img)


###################### Run Machine Learning Method ######################
if algo == 'NB':
  # Naive Bayes runs very fast
  print("\nRunning Naive Bayes......\n")
  test_predictions, acc, ll = ML_Methods.run_naive_bayes(train, test, ss_split, labels)
elif algo == 'SVM':
  print("\nRunning Support Vector Machine......\n")
  # SVM takes quite awhile to get the final results (5 to 10 mins)
  test_predictions, acc, ll = ML_Methods.run_support_vector_machine(train, test, ss_split, labels)
elif algo == 'LR':
  print("\nRunning Logistic Regression......")
  # Logistic Regression takes around 3-5 mins to get the final results
  test_predictions, acc, ll = ML_Methods.run_logistic_regression(train, test, ss_split, labels)
elif algo == 'KNN':
  print("\nRunning Logistic K-nearest Neighbours......\n")
  # K-nearest Neighbours runs very fast
  test_predictions, acc, ll = ML_Methods.run_k_nearest_neighbours(train, test, ss_split, labels)
elif algo == 'LDA':
  print("\nRunning Linear Discriminant Analysis......\n")
  # Linear Discriminant Analysis runs very fast
  test_predictions, acc, ll = ML_Methods.run_linear_discriminant_analysis(train, test, ss_split, labels)
else:
  print("\nPlease provide a ML algorithm to run.\n")

###################### Postprocessing Data ######################
# Print results
print ('Accuracy: {:.4%}'.format(acc))
print ('Log Loss: {:.6}'.format(ll))

# Format DataFrame
submission = pd.DataFrame(test_predictions, columns=classes)
submission.insert(0, 'id', test_ids)
submission.reset_index()

# Export Submission
submission.to_csv('submission.csv', index = False)

# Double check the output
submission.head()
