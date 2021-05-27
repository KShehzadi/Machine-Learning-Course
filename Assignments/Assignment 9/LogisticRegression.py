# import the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# import the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# get dummy variables
dummy_dataset=pd.get_dummies(data=dataset, columns=['Gender'])

# seperate X and y variables
X = dummy_dataset.drop('Purchased',axis=1)
y = dummy_dataset['Purchased']

# split the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# fit Logistic Regression to the training set
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# predict the Test set results
y_pred = classifier.predict(X_test)

# make the confusion matrix for evaluation
print("confusion matrix:")
con_matrix = confusion_matrix(y_test, y_pred)
print(con_matrix)
# calculating total data points
total_datapoints = con_matrix[0][0] + con_matrix[1][1] + con_matrix[0][1] + con_matrix[1][0]
print("Total data points: " , total_datapoints)
# calculating correctly predicted data points ratio
correct_preds = con_matrix[0][0] + con_matrix[1][1]
print("Correct predictions:", correct_preds)
# calculating the datapoints ratio which are incorrectly predicted
incorrect_preds = con_matrix[0][1] + con_matrix[1][0]
print("Incorrect predictions:" , incorrect_preds)
# calculating the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)