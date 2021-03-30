import csv
import time

#function to take input of start and end range from user
def readStartandEndRange():
    startRange = input("Enter start range for model prediction: ")
    endRange = input("Enter end range for model prediction: ")
    return startRange, endRange

#function to load data from a file to a 2d list data strcture. Takes file name as input
def readFileAndLoadData(filename):
    dataList = []
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            dataList.append((row[0], row[1]))

    return dataList

#function to calculate Loss in a row for selected model
def calculateLoss(yRealValue, m, x, c):
    return (yRealValue - ((m*x) + c))**2

#function to calculate cost of predicted(m, c) or theeta1 and theeta0 in given dataList
def calculateCost(dataList, predictedM, predictedC):
    cost = 0
    for dataPoint in dataList:
        cost = cost + calculateLoss(int(dataPoint[1]), predictedM, int(dataPoint[0]), predictedC)
    return cost/(2*len(dataList))

#function to calculate minimum cost model in given startRange and endRange for dataList
def findMinimumCostAndBestModel(startRange, endRange, dataList):
    minimumCost= 1000000000000000000000000000
    minimumM = 0
    minimumC = 0
    numberofIterations= 0
    for m in range(startRange, endRange+1):
        for c in range(startRange, endRange+1):
            numberofIterations += 1
            cost = calculateCost(dataList, m, c)
            print('Model: ( 𝜃1: ', m, ' ,𝜃0: ', c, ') Cost:' ,cost)
            if(cost < minimumCost):
                minimumCost = cost
                minimumM = m
                minimumC = c
    return (minimumM, minimumC), numberofIterations


#Driver Program
print('*******************************************************Selection of Start and End Range***************************************************************************')
#taking start and end range input from user
startRange, endRange = readStartandEndRange()
print('******************************************************************************************************************************************************************')
# starting time
start = time.time()
#load data from file to dataList data structure
dataList = readFileAndLoadData('data.csv')
print('**************************************************************************************************************************************************************')
print('*******************************************************DATA POINTS LOADED FROM FILE***************************************************************************')
#printing dataList loaded from file
print(dataList)
print('**************************************************************************************************************************************************************')
print('')
print('***********************************************************Processing All Models***********************************************************************************')
#calculating best model in given range and loaded data list
bestModel, numberofIterations = findMinimumCostAndBestModel(int(startRange), int(endRange),dataList)
print('********************************************Best univariate linear model for given data points in user defined range**************************************************')
#printing best model values
print('( 𝜃1: ',bestModel[0], ', 𝜃0: ', bestModel[1], ')')
print('******************************************************************************************************************************************************************')
# end time
end = time.time()
# total time taken
print('*********************************************************************Runtime and Iterations Count Of Algorithm************************************************************************')
#printing runtime and number of iterations performed by algorithm 
print(f"Runtime of the program is {end - start} seconds")
print(f"number of iterations taken is {numberofIterations}")
print('******************************************************************************************************************************************************************')