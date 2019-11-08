import tensorflow as tf
from tensorflow import keras
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import pickle

data = pd.read_csv("breast-cancer.data", sep =",")
#print(data.head())
predictionMap = {'no-recurrence-events': 0,
                 'recurrence-events': 1}
prediction = []
for x in data["class"]: prediction.append(predictionMap[x])
data = data[["age", "menopause", "tumor-size", "breast", "node-caps", "irradiat"]]
#print(data.head())
#ageS = [int(sub.split('-')[1]) for sub in data["age"]]
ageS = []
ageE = []
mp = []
mpp = {'premeno': 0,
       'ge40': 1,
       'lt40':2}
tsm = [] #tumor size mean value
brst = []
breastMap = {'left': 0,
             'right': 1}
nc = []
ncMap ={'?': 0,
        'no': 1,
        'yes': 2}
irr = []
for x in data["age"]:
    ageS.append((int)(x.split('-')[0]))
    ageE.append((int)(x.split('-')[1]))
for x in data["menopause"]: mp.append(mpp[x])
for x in data["tumor-size"]:
    temp = x.split('-')
    meanVal =((int)(temp[0]) + (int)(temp[1]))/2
    tsm.append(meanVal)
for x in data["breast"]: brst.append(breastMap[x])
for x in data["node-caps"]: nc.append(ncMap[x])
for x in data["irradiat"]: irr.append(ncMap[x])
#print(mp)
dataframe = {'StartAge': ageS,
             'EndAge': ageE,
             'Menopause': mp,
             'TumorSize': tsm,
             'Breast': brst,
             'NodeCap': nc,
             'Irradiat': irr}
newData = pd.DataFrame(dataframe)

x = np.array(newData)
#print(x)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, prediction, test_size = 0.3)

"""print(x_train)
print(y_train)
print(len(y_train))
print(x_test)
print(y_test)
print(len(y_test))"""

#creating model
model = sklearn.linear_model.LogisticRegression(class_weight="balanced", solver="liblinear", penalty="l2")

model.fit(x_train, y_train)
predicted = model.predict(x_test)
score = model.score(x_test, y_test)
print("Accuracy: " ,score)

pickle_in = open("bestModel.pickle", "rb")
comparable = pickle.load(pickle_in)

#replacing the current model if it has higher accuracy than the best one yet
threshold = comparable.score(x_test, y_test)
print("Prev acc: ", threshold)
if(score > threshold):
    with (open("bestModel.pickle", "wb")) as f:
        pickle.dump(model, f)
    print("Model saved.")

#visualising model's mistakes
"""for x in range(len(predicted)):
    print(predicted[x], x_test[x], y_test[x])"""


