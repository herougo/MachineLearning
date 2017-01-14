# alternate interface for matplotlib to display plots with one
# line of python

# To do: account for multiple lines in the same graph
# variables: colours, filled_markers, default_marker

# plt.plot([1,2,3,4], [1,4,9,16], 'ro')
# plt.axis([x_start, x_end, y_start, y_end])

# ax.set_aspect(2)

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

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
