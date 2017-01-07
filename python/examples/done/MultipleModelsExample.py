# try many models on small training set and print the model with the best accuracy

from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron

# any time you want to add new models, just add them to the dictionary!
clf = {
    'DecisionTree': tree.DecisionTreeClassifier(), 
    'SVM': svm.SVC(),
    'GaussianNB': GaussianNB(),
    'RandomTree': RandomForestClassifier(n_estimators=100),
    'KNeighbors': KNeighborsClassifier(),
    'Perceptron': Perceptron()
}

accuracy = {}

#training data
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
'female', 'male', 'male']

#test data

# train and get accuracy of each model
for model in clf.keys():
    clf[model] = clf[model].fit(X, Y)
    prediction = clf[model].predict(X)
    accuracy[model] = accuracy_score(Y, prediction)

# print models in descending order wrt accuracy
for key, value in sorted(accuracy.items(), key=lambda kv: kv[1], reverse=True):
    print(key, value)

# get key of best model
best = max(accuracy, key=accuracy.get)
print("Best gender classifier is", best, "with accuracy", accuracy[best])

