# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from support import *

Y = np.array([52  ,60   ,67   ,70  ,72  ,90  ,80  ,85  ,99  ,110 ,108,100])
X = np.array([1.45, 1.60, 1.65,1.70,1.75,1.80,1.85,1.90,1.92,1.93,2.0,2.1])


def gradient_descent(x_values_original, y_values_original, iterations = 1000,
                      learning_rate = 0.001, stopping_threshold = 1e-6):
    """Walk the curve to find the prediction-line"""
    k = 0.1
    m = 0.01
    nr_of_values = len(x_values_original)
    # Save values for later use
    list_of_errors = []
    list_of_k_values = []

    previous_mean_squared_error = None

    for i in range(iterations):
        y_predicted_values = k * x_values_original + m
        current_mean_squared_error = mean_squared_error(y_values_original, y_predicted_values)

        # CHeck if we've reached stopping_threshold. If so: Break the loop
        if previous_mean_squared_error != None and \
            abs(previous_mean_squared_error - current_mean_squared_error) <= stopping_threshold:
            break
        # if not: store this error as previous
        previous_mean_squared_error = current_mean_squared_error

        # Save this for later use, when we plot it
        list_of_errors.append(current_mean_squared_error)
        list_of_k_values.append(k)

        # Calculate the gradients - 
        k_gradient = -2 * np.mean((y_values_original - y_predicted_values) * x_values_original)
        m_gradient = -2 * np.mean(y_values_original - y_predicted_values)

        # Adjust the line and re-check
        k = k - (learning_rate * k_gradient)
        m = m - (learning_rate * m_gradient)

    plot_mean_error(list_of_k_values, list_of_errors)

    return k, m


# Estimating line slope and intercept using gradient descent
k, m = gradient_descent(X,Y, iterations=10000, learning_rate=0.001)

Y_pred = k*X+m

plot_regression_line(X, Y, Y_pred)

print("A person who is 2.3 meters would weigh: " + str(k*2.3+m) + " kg")






    
    
