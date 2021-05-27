import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


def ComputeCost(X,y,theta):
    temp = np.dot(X,theta) - y
    return np.sum(np.power(temp,2)) / (2*m)

def plotTheetaGraph(theetalist, costs, xAxisLabel):
    plt.plot(theetalist, costs)
    plt.xlabel(xAxisLabel)
    plt.ylabel("Cost function  J(Θ)")
    plt.title("Gradient Descent")
    plt.show()



def GradientDescentMulti(X,y,theta,alpha,iterations):
    m = len(y)
    theta0list = []
    theta1list = []
    theta2list = []
    costs = []
    for _ in range(iterations):
        temp = np.dot(X,theta) - y
        temp = np.dot(X.T,temp)
        theta = theta - ((alpha/m) * temp)
        theta0list.append(theta[0][0])
        theta1list.append(theta[1][0])
        theta2list.append(theta[2][0])
        costs.append(ComputeCost(X, y, theta))
    return theta, theta0list, theta1list, theta2list, costs



def predict(predict_data):
    prediction = np.dot(predict_data,theta)
    return prediction
def printActualandPredictedValues(actualprice, predictedprice):
    for i in range(len(actualprice)):
        print("Actual: ", actualprice[i], " Predicted: ", predictedprice[i][0]);

def writeToThetaFile(filename, thetalist, costs, thetavalue, alpha):
     with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([thetavalue, "Cost Function"])
        for i in range(len(thetalist)):
             writer.writerow([thetalist[i], costs[i]])
        writer.writerow(["Learning Rate", alpha])

def writeToPredictionFile(filename, X, y, preds, least_square_errors, cost):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["X-Value", "Actual-Value", "Predicted-Value", "Least Square Error"])
        for i in range(len(preds)):
             writer.writerow([X[i][0], y[i], preds[i][0], least_square_errors[i]])
        writer.writerow(["Average Error", cost])
def calculate_lease_square_errors(price, actualprice):
    least_square_errors = []
    for i in range(len(actualprice)):
        least_square_errors.append((price[i][0]-actualprice[i])**2)
    return least_square_errors

data = pd.read_csv('simple.csv',sep = ',', header = None) #loading data from the file

X = data.iloc[:,0:2] #read first two columns into X
y = data.iloc[:,2]   #read the third column into y

m = len(y) #total number of training samples

data.head()
#normalize data
X = (X-np.mean(X))/np.std(X)

ones = np.ones((m,1))

X = np.hstack((ones,X)) #adding the intercept term

alpha = 0.01
iterations = 10000

theta = [[0],[0],[0]]

y = y[:,np.newaxis]

J = ComputeCost(X,y,theta)
print("Cost with initial theta values: ", J)

theta, theta0list, theta1list, theta2list, costs = GradientDescentMulti(X,y,theta,alpha,iterations)


print("Computed theta values", theta)

J = ComputeCost(X,y,theta)
print("Cost with computed theta values: ", J)

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
least_square_errors = calculate_lease_square_errors(price, actualprice)
cost = np.sum(least_square_errors)/(2*m_predict)
writeToPredictionFile('Predictions.csv', predict_data, actualprice, price, least_square_errors, cost)
writeToThetaFile("CostFunction_Theta0.csv", theta0list, costs, "Theta0_Value", alpha)
writeToThetaFile("CostFunction_Theta1.csv", theta1list, costs, "Theta1_Value", alpha)
writeToThetaFile("CostFunction_Theta2.csv", theta2list, costs, "Theta1_Value", alpha)
plotTheetaGraph(theta0list, costs, 'theeta 0 alpha:'+ str(alpha))
plotTheetaGraph(theta1list, costs, 'theeta 1 alpha:'+ str(alpha))
plotTheetaGraph(theta2list, costs, 'theeta 2 alpha:'+ str(alpha))