import numpy as np
import pylab as P
import pandas as pd

def exp(x):
    return np.exp(x)

# Unnecessary (sum works with arrays and numpy arrays
#def sum(x):
#    return np.sum(x, axis=0)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) /  np.sum(np.exp(x))

# get training data (train.csv/test.csv or train/test folders or data folder)

# print data (and noteworthy graphs and results from models)

# clean data
# - fill in missing numbers with median (or most common element) 

# add features

# retrieve training data (minus labels) as np array

# retrieve training data classification column as np array

# train

# train multiple models

# predict



# write predictions to csv file
import csv as csv
# ie savePredictions("myfirst.csv", "Id", "Survived", ids, output)
def savePredictions(filename, id_label, output_label, ids, output):
    predictions_file = open(filename, "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow([id_label, output_label])
    open_file_object.writerows(zip(ids, output))
    predictions_file.close()

''' *********************************************************
***************** Helper Module Ideas ***********************
*************************************************************
data helper which can interface with pandas or MongoDB




''