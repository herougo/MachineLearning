#1
import numpy as np
import pylab as P
import pandas as pd

'''
import csv as csv 

# Open up the csv file in to a Python object
csv_file_object = csv.reader(open('C:/Henri/My Documents/1 - Priorities/Unsorted/Kaggle/Titanic/train.csv', 'rb')) 
header = csv_file_object.next()  # The next() command just skips the first line which is a header
data=[]                          # Create a variable called 'data'.
for row in csv_file_object:      # Run through each row in the csv file,
    data.append(row)             # adding each row to the data variable
data = np.array(data)              # Then convert from a list to an array

print(data)



test_file = open('../csv/test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

prediction_file = open("genderbasedmodel.csv", "wb")
prediction_file_object = csv.writer(prediction_file)

prediction_file_object.writerow(["PassengerId", "Survived"])
for row in test_file_object:       # For each row in test.csv
    if row[3] == 'female':         # is it a female, if yes then                                       
        prediction_file_object.writerow([row[0],'1'])    # predict 1
    else:                              # or else if male,       
        prediction_file_object.writerow([row[0],'0'])    # predict 0
test_file.close()
prediction_file.close()

'''

'''
#np.sum(..)
#np.size(..)
#np.mean(..)
#np.unique()
#np.arange(0., 5., 0.2) -> [0., 0.2, 0.4, ... 5.
#np.zeros(4,float) (array of 4 zeroes)
#np.zeroes([4, 2], float (array of 4 arrays of 2 zeroes each)
r1 = [ '0.1', '1.1', '1.5']
r2 = [ '2.2', '3.2', '3.5']
data = []
data.append(r1)
data.append(r2)
data = np.array(data)
# data = data.astype(np.float)
# fail: data[0::, 0] = data[0].astype(np.float)

mask = data[0::, 1] == '1.1'
# [ True False ]
masked_rows = data[mask, 0::].astype(np.float)
print(masked_rows)

print((data[0::, 1] == '1.1').astype(np.int))
# print 1s and 0s instead of true and false 

print(data)
print(data[0::,1])
print(data[0::,0].astype(np.float))
print(data[0,::1].astype(np.float))
print(data[0].astype(np.float))

data[data[0::, 0] == '0.1', 0::] = '0'
print(data)


survival_table = np.zeros([4, 2],float)
survival_table = [[1, 2, 3],
                  [4, 5, 6]]
survival_table = np.array(survival_table)
print(survival_table % 2 == 1)
survival_table[ survival_table % 2 == 1 ] = 1 


print(survival_table)


def yo():
    return [1, 2]

def yo2():
    return (1, 2)

def yo3():
    return 1, 2

hi, there = yo()
hi, there = yo2()
hi, there = yo3()

for kv in accuracy.items():
    print(kv, kv[0], kv[1])
    
for kv in accuracy:
    print(kv, kv[0], kv[1])
'''

#2 **********************************************

def test1():
    # For .read_csv, always use header=0 when you know row 0 is the header row
    df = pd.read_csv('../train.csv', header=0)
    
    #print(df.head(3))
    #print(df.dtypes)
    #df.info()
    #df.describe()
    
    print(df['Age'][0:10])
    
    df[df['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']]
    
    for i in range(1,4):
        print(i, len(df[ (df['Sex'] == 'male') & (df['Pclass'] == i) ]))
        
    df['Age'].hist()
    #P.show()

    print(df.head(3)[['Age', 'Cabin']])
    print(df.head(3).dropna()[['Age', 'Cabin']])
    
    df['Gender'] = df['Sex'].map( lambda x: x[0].upper() )
    df.Sex = df.Sex.map( {'female': 0, 'male': 1} ).astype(np.int)
    
    print("YAAY")
    
    print(df[df.columns[5]].head(5))
    print(df['Age'].head(5))
    
    df[ df['Age'].isnull() ]
    
    #for i in range(0, 2):
    #    for j in range(0, 3):
    #         median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()
    
    print(df.dtypes[df.dtypes.map(lambda x: x=='object')])

def exp(x):
    return np.exp(x)

# Unnecessary (sum works with arrays and numpy arrays
#def sum(x):
#    return np.sum(x, axis=0)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # pass  # TODO: Compute and return softmax(x)
    return np.exp(x) /  np.sum(np.exp(x))

def test2():
    scores = np.array([3.0, 1.0, 0.2])
    print(softmax(scores * 10)) # values approach 1 or 0
    
    print(softmax(scores / 10)) # values approach uniform distribution

def test3():
    print("Numerical instability (ie adding big numbers to small numbers)")
    n = 1000 * 1000 * 1000
    for i in range(0, 1000 * 1000):
        n += 0.000001
    n -= 1000 * 1000 * 1000
    print(n)
    print("Should be 1")
