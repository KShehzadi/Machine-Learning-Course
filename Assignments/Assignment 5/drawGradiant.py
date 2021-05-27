import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
#take function as an input like x**2 + y**2 and parse it using lambda function
def inputFunction():
    print("Enter the function expression, e.g. x**2+4*y")
    x, y = sym.Symbol('x'), sym.Symbol('y')
    stringFunc = input()
    lambdaFunc = lambda x, y: eval(stringFunc)
    return lambdaFunc, stringFunc
#find gradiant vector of a function general and on a given point
def findGradiantVector(stringFunc, xPoint, yPoint):
    x, y = sym.Symbol('x'), sym.Symbol('y')
    p_derivative_x, p_derivative_y = sym.diff(stringFunc, x), sym.diff(stringFunc, y)
    print('Gradiant Vector for given function: [',p_derivative_x,',' ,p_derivative_y,']')
    p_derivative_x_value, p_derivative_y_value = lambda x, y:eval(str(p_derivative_x)), lambda x, y:eval(str(p_derivative_y))
    return p_derivative_x_value(int(xPoint),int(yPoint)),p_derivative_y_value(int(xPoint),int(yPoint))

#take a 2 point input from user
def inputPoint():
    x, y = input("Enter the point on which gradiant vector needs to be calculated, e.g 1,2 : ").split(",")
    return x, y
#draw graph of the userdefined input
def drawGraph(lambdaFunc, xPoint, yPoint, xGradiant, yGradiant):
    x, y = np.linspace( xPoint-1000,  xPoint+1000, 100), np.linspace(yPoint-1000, yPoint+1000, 100)
    X, Y = np.meshgrid(x, y)
    Z = lambdaFunc(X, Y)
    ax = plt.axes(projection ='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='viridis',edgecolor='none')
    ax.set_title('Graph of given function')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.plot(xGradiant,yGradiant, 'go')
    plt.plot(xGradiant*2,yGradiant*2, 'bo')
    plt.show()
      
if __name__ == "__main__":
    lambdaFunc , stringFunc = inputFunction()
    xPoint, yPoint = inputPoint()
    xGradiant, yGradiant = findGradiantVector(stringFunc, xPoint, yPoint)
    print('Gradiant Vector on given points:[',xGradiant,',', yGradiant ,']')
    drawGraph(lambdaFunc, int(xPoint), int(yPoint), xGradiant, yGradiant)