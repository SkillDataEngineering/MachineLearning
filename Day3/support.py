import matplotlib.pyplot as plt
import numpy as np

def plot_mean_error(k_values, error_values):
    plt.figure(figsize=(8,6))
    #plt.plot(k_values, error_values)
    plt.scatter(k_values, error_values, marker='o', color='red')
    plt.title("Mean-Error for each k-value")
    plt.ylabel("Mean-error")
    plt.xlabel("k-value")
    plt.show()

def plot_regression_line(X, Y, Y_pred):
    plt.figure(figsize = (8,6))
    plt.scatter(X,Y, marker ='o', color='red')
    plt.plot(X,Y_pred, color='blue', linestyle='dashed')
    plt.xlabel("Height in meters")
    plt.ylabel("Weight in kilograms")
    plt.title("Linear regression")
    plt.show()

def squared_error(y1,y2):
    """Finds the y-distance between two points with the same x-value
    and returns it's square"""
    return (y1-y2)**2

def mean_squared_error(y_values_original, y_predicted_values):
    """The y-values are the original ones and the predicted y are from
    our current curve which we want to test the mean_squared_error for"""
    nr_of_values = len(y_values_original)
    sum_of_errors = 0
    # Loop through all values
    for i in range(nr_of_values):
        sum_of_errors += squared_error(y_values_original[i],y_predicted_values[i])
        
    # Lastly, we want to return the mean_value of errors
    return sum_of_errors / nr_of_values