import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import csv
def readFileAndLoadData(filename):
    X = []
    y = []
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            X.append((row[0], row[1]))
            y.append(row[1])

    return X, y
def plotTheetaGraph(theetalist, costs, xAxisLabel):
    plt.plot(theetalist, costs)
    plt.xlabel(xAxisLabel)
    plt.ylabel("Cost function  J(Θ)")
    plt.title("Gradient Descent")
    plt.show()

def readData(filename):
    X, y = readFileAndLoadData(filename);
    X = X[1: len(X)];
    y = y[1 : len(y)];
    X = np.array(X).astype('float64')
    y = np.array(y).astype('float64')
    return X, y
def gradPredict(X,theta):
    preds = []
    for x in X:
        preds.append((x[0]*theta[1]) + theta[0])
    return preds

def calculateCost(X,y,theta):
    preds = gradPredict(X,theta)
    return (abs(preds-y)).mean()

def gradDesecent(X,y,alpha=0.05,maxIter=30000):
    thetas = []
    theeta0list = []
    theeta1list= []
    costs = [];
    i = 0
    theta = np.zeros(2, dtype=np.float64)
    theta = np.array([0,0])
    converged=False
    while converged==False:
        preds = gradPredict(X,theta)
        theta0 = theta[0] - alpha*(preds-y).mean()
        theta1 = theta[1] - alpha*((preds-y)*X[:,1]).mean()
        theta = np.array([theta0,theta1])
        J = calculateCost(X,y,theta)
        if len(costs)> 0 and abs(costs[len(costs)-1]-J) <= 0.0001:
            converged = True
        if maxIter == i:
            converged = True
        theeta0list.append(theta0)
        theeta1list.append(theta1)
        thetas.append(theta)
        costs.append(J)
        i += 1
    validcosts = np.isfinite(costs)
    minIndex = np.argmin(validcosts) - 1;
    print(f'theta: {thetas[minIndex]} | cost: {costs[minIndex]} | iteration: {i}  ')
    return thetas[minIndex], theeta0list, theeta1list, costs 


def calcualteModelandPlotonTrainData():
    filename = input("Enter the name of train data file: ")
    X, y = readData(filename)
    alpha = input("Enter the value of learning rate: ")
    thetas, theeta0list, theeta1list, costs = gradDesecent(X,y, float(alpha))
    plotTheetaGraph(theeta0list, costs, 'theeta 0 alpha:'+alpha)
    plotTheetaGraph(theeta1list, costs, 'theeta 1 alpha:'+alpha)
    return thetas, alpha, theeta0list, theeta1list, costs

def calculateLossOnTestData(thetas):
    filename = input("Enter the name of test data file: ")
    X, y = readData(filename)
    cost = calculateCost(X, y, thetas)
    preds = gradPredict(X,thetas)
    least_square_errors = (preds-y)**2
    cost = least_square_errors.mean()/(2)
    writeToPredictionFile('Predictions.csv', X, y, preds, least_square_errors, cost)
    return
def writeToPredictionFile(filename, X, y, preds, least_square_errors, cost):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["X-Value", "Actual-Value", "Predicted-Value", "Least Square Error"])
        for i in range(len(preds)):
             writer.writerow([X[i][0], y[i], preds[i], least_square_errors[i]])
        writer.writerow(["Average Error", cost])
def writeToThetaFile(filename, thetalist, costs, thetavalue, alpha):
     with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([thetavalue, "Cost Function"])
        for i in range(len(thetalist)):
             writer.writerow([thetalist[i], costs[i]])
        writer.writerow(["Learning Rate", alpha])
def main():
    thetas, alpha, theeta0list, theeta1list, costs = calcualteModelandPlotonTrainData()
    calculateLossOnTestData(thetas)
    writeToThetaFile("CostFunction_Theta0.csv", theeta0list, costs, "Theta0_Value", alpha)
    writeToThetaFile("CostFunction_Theta1.csv", theeta1list, costs, "Theta1_Value", alpha)
if __name__ == '__main__':
    main();
    