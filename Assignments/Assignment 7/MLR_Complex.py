import numpy as np
import pandas as pd

data = pd.read_csv('complex_train.csv',sep = ',', header = None) #loading data from the file

X = data.iloc[:,0:2] #read first two columns into X
y = data.iloc[:,2]   #read the third column into y

m = len(y) #total number of training samples

data.head()

X = (X-np.mean(X))/np.std(X)

ones = np.ones((m,1))

X = np.hstack((ones,X)) #adding the intercept term

alpha = 0.01
iterations = 400

theta = np.zeros((3,1))

y = y[:,np.newaxis]

def ComputeCost(X,y,theta):
    temp = np.dot(X,theta) - y
    return np.sum(np.power(temp,2)) / (2*m)

J = ComputeCost(X,y,theta)
print("Cost with initial theta values: ", J)

def GradientDescentMulti(X,y,theta,alpha,iterations):
    m = len(y)
    thetas = []
    costs = []
    for _ in range(iterations):
        temp = np.dot(X,theta) - y
        temp = np.dot(X.T,temp)
        theta = theta - ((alpha/m) * temp)
        thetas.append((theta[0][0],theta[1][0]))
        costs.append(ComputeCost(X, y, theta))
    
    return theta, thetas, costs

theta, thetas, costs = GradientDescentMulti(X,y,theta,alpha,iterations)

print("Computed theta values", theta)

J = ComputeCost(X,y,theta)
print("Cost with computed theta values: ", J)

def predict(predict_data):
    prediction = np.dot(predict_data,theta)
    return prediction
def printActualandPredictedValues(actualprice, predictedprice):
    for i in range(len(actualprice)):
        print("Actual: ", actualprice[i], " Predicted: ", predictedprice[i][0]);

predict_data = pd.read_csv('simple.csv',sep = ',', header = None)
actualprice = predict_data.iloc[:,2]  
predict_data = predict_data.iloc[:,0:2]


predict_data = (predict_data-np.mean(predict_data))/np.std(predict_data) #feature normalization

m_predict = len(predict_data)

ones = np.ones((m_predict,1))
print("Predicting prices for test data.")
predict_data = np.hstack((ones,predict_data)) #adding the intercept term

price = predict(predict_data)

printActualandPredictedValues(actualprice, price)