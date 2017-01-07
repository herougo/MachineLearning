import matplotlib.pyplot as plt
import pandas as Series

def formatArray(x):
	if type(x) == pd.Series:
		x = x.values
	return x

# histogram
def hist():
    return ""

def scatterplot(x_values, y_values, x_label="", y_label="", title=""):
    plt.plot(x_values, y_values, 'ro')
    if title != "":
        plt.title(title)
    if x_label != "":
        plt.xlabel = x_label
    if y_label != "":
        plt.ylabel = y_label
    plt.show()
    

# line graph
def linegraph(x_values, y_values, x_label="", y_label="", title=""):
    plt.plot(x_values, y_values)
    if title != "":
        plt.title(title)
    if x_label != "":
        plt.xlabel = x_label
    if y_label != "":
        plt.ylabel = y_label
    plt.show()
