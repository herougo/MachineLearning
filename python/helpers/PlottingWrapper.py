# alternate interface for matplotlib to display plots with one
# line of python

# To do: account for multiple lines in the same graph

import matplotlib.pyplot as plt
import pandas as pd

# Purpose: if a panda series is passed, format it to be compatible
#          with matplotlib
def formatArray(x):
	if type(x) == pd.Series:
		x = x.values
	return x

# histogram
def histogram():
    return ""

# box and whisker plot
def boxplot():
	return ""

# confusion matrix as image
def confusionmatrix():
	return ""

# plot signals given matrix
def signals():
	return ""

# scatter plot
def scatter(x_values, y_values, x_label="", y_label="", title=""):
    x_values = formatArray(x_values)
    y_values = formatArray(y_values)
    plt.plot(x_values, y_values, 'o')
    if title != "":
        plt.title(title)
    if x_label != "":
        plt.xlabel = x_label
    if y_label != "":
        plt.ylabel = y_label
    plt.show()
    

# line plot
def line(x_values, y_values, x_label="", y_label="", title=""):
    x_values = formatArray(x_values)
    y_values = formatArray(y_values)
    plt.plot(x_values, y_values)
    if title != "":
        plt.title(title)
    if x_label != "":
        plt.xlabel = x_label
    if y_label != "":
        plt.ylabel = y_label
    plt.show()
