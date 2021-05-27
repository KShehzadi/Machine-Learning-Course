def drawDecisionBoundary(features,targets,classifier,Title=""):
    from matplotlib.colors import ListedColormap
    X_set, y_set = features, targets
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                        np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Logistic Regression ({0})'.format(Title))
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

   

# Logistic Regression
#Udemy Code GitHub
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

print(X[2:])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics

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
# Visualising the Training set results
drawDecisionBoundary(X_train, y_train,classifier)
# Visualising the Test results
#drawDecisionBoundary(X_test, y_test,classifier,Title="TestSet")