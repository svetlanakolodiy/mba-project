# Pandas is used for data manipulation
import pandas as pd
# Use numpy to convert to arrays
import numpy as np
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


# Read in data and display first 5 rows
features = pd.read_csv('ffp_train.csv')
print(features.head(5))

print('The shape of our features is:', features.shape)

# Descriptive statistics for each column
print(features.describe())


# Labels are the values we want to predict
labels = np.array(features['BUYER_FLAG'])
# Remove the labels from the features
# axis 1 refers to the columns
features = features.drop('BUYER_FLAG', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)


# # create a base classifier used to evaluate a subset of attributes
# lr = LogisticRegression()
# # create the RFE model and select 150 attributes
# rfe = RFE(lr, 150)
# rfe = rfe.fit(features, labels)
# # summarize the selection of the attributes
# print(rfe.support_)
# print(rfe.ranking_)

# fit an Extra Trees model to the data
etc = ExtraTreesClassifier()
etc.fit(features, labels)
# display the relative importance of each attribute
print(etc.feature_importances_)

# Get numerical feature importances
importances = list(etc.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
print('\n'.join('{}: {}'.format(*pair) for pair in feature_importances))


# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# Instantiate model with 1000 decision trees (consider larger n_estimator (1000))
rf = RandomForestRegressor(n_estimators = 10, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)


# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

accuracy = accuracy_score(test_labels, (predictions>0.5).astype(int))
print(accuracy)
precision = precision_score(test_labels, (predictions>0.5).astype(int))
print(precision)
recall = recall_score(test_labels, (predictions>0.5).astype(int))
print(recall)
f1 = f1_score(test_labels, (predictions>0.5).astype(int))
print (f1)






