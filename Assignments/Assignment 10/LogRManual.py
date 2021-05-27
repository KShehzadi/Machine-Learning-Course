

# import the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from math import exp
import matplotlib.pyplot as plt
# Creating the logistic regression model

# Helper function to normalize data
def normalize(X):
    return X - X.mean()

# Method to make predictions
def predict(X, b0, b1):
    return np.array([1 / (1 + exp(-1*b0 + -1*b1*x)) for x in X])


def ComputeCost(X,y,theta):
    temp = np.dot(X,theta) - y
    return np.sum(np.power(temp,2)) / (2*m)
# Method to train the model
def logistic_regression(X, Y):

    X = normalize(X)

    # Initializing variables
    b0 = 0
    b1 = 0
    L = 0.001
    epochs = 300
    costs = []
    b0_list = []
    b1_list = []
    for epoch in range(epochs):
        y_pred = predict(X, b0, b1)
        cost = sum(Y - y_pred)/len(X)
        D_b0 = -2 * sum((Y - y_pred) * y_pred * (1 - y_pred))  # Derivative of loss wrt b0
        D_b1 = -2 * sum(X * (Y - y_pred) * y_pred * (1 - y_pred))  # Derivative of loss wrt b1
        b0_list.append(b0)
        b1_list.append(b1)
        # Update b0 and b1
        b0 = b0 - L * D_b0
        b1 = b1 - L * D_b1
        costs.append(cost)
    
    return b0, b1, b0_list, b1_list, costs

def plotTheetaGraph(theetalist, costs, xAxisLabel):
    plt.plot(theetalist, costs)
    plt.xlabel(xAxisLabel)
    plt.ylabel("Cost function  J(Θ)")
    plt.title("Gradient Descent")
    plt.show()



plt.rcParams["figure.figsize"] = (10, 6)

# Load the data
data = pd.read_csv("Social_Network_Ads.csv")
data.head()

plt.scatter(data['Age'], data['Purchased'])
plt.show()

# Divide the data to training set and test set
X_train, X_test, y_train, y_test = train_test_split(data['Age'], data['Purchased'], test_size=0.20)

b0, b1, b0_list, b1_list, costs= logistic_regression(X_train, y_train)

print("Weights", b0, ",", b1);
plotTheetaGraph(b0_list, costs, "Weight 0")

plotTheetaGraph(b1_list, costs, "Weight 1")
# Making predictions
X_test_norm = normalize(X_test)
y_pred = predict(X_test_norm, b0, b1)
y_pred = [1 if p >= 0.5 else 0 for p in y_pred]

plt.clf()
plt.scatter(X_test, y_test)
plt.scatter(X_test, y_pred, c="red")
plt.show()


cm = confusion_matrix(y_test, y_pred,labels=[1,0])
print("Confusion Matrix:")
print(cm)
TP, FN, FP, TN = cm.reshape(-1)
accuracy = (TP + TN) / float(TP + TN + FP + FN)
print("Accuracy: " ,accuracy )
recall = (TP) / float(TP + FN)
print("Recall: " ,recall )
precision = (TP) / float(TP + FP)
print("Precision: " , precision)
f1_measure = (2*precision*recall)/float(precision + recall)
print("F1 measure: " , f1_measure)
specificity = (TN) / float(TN + FP)
print("Specificity: " ,specificity )