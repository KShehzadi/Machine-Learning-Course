
import numpy as np

class LinearRegressor:
    def __init__(self, x, y, alpha = 0.01, b0 = 0, b1 = 0):
        """ 
            x: input feature
            y: result / target
            alpha: learning rate, default is 0.01
            b0, b1: linear regression coefficient.
        """
        self.i = 0
        self.x = x
        self.y = y
        self.alpha = alpha
        self.b0 = b0
        self.b1 = b1
        if len(x) != len(y):
            raise TypeError("x and y should have same number of rows.")
  
    def predict(model, x):
        """Predicts the value of prediction based on 
           current value of regression coefficients when input is x"""
        # Y = b0 + b1 * X
        return model.b0 + model.b1 * x
  
    def cost_derivative(model, i):
        x, y, b0, b1 = model.x, model.y, model.b0, model.b1
        predict = model.predict
        return sum([
            2 * (predict(xi) - yi) * 1
            if i == 0
            else (predict(xi) - yi) * xi
            for xi, yi in zip(x, y)
        ]) / len(x)
  
    def update_coeff(model, i):
        cost_derivative = model.cost_derivative
        if i == 0:
            model.b0 -= model.alpha * cost_derivative(i)
        elif i == 1:
            model.b1 -= model.alpha * cost_derivative(i)
  
    def stop_iteration(model, max_epochs = 1000):
        model.i += 1
        if model.i == max_epochs:
            return True
        else:
            return False
  
    def fit(model):
        update_coeff = model.update_coeff
        model.i = 0
        while True:
            if model.stop_iteration():
                break
            else:
                update_coeff(0)
                update_coeff(1)
  
  
if __name__ == '__main__':
    linearRegressor = LinearRegressor(
        x =np.array([1, 2, 3, 4, 5, 6, 7, 0.0]),
        y = np.array([2, 3, 4, 5, 6, 7, 8, 0.0]),
        alpha = 0.5
    )
    linearRegressor.fit()
    print(linearRegressor.predict(12))
  
    # expects 2 * 12 + 3 = 27