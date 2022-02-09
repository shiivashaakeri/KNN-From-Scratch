from math import sqrt
import pandas as pd
import numpy as np
from collections import Counter


def trainTestSplit(data):
    rowNum = data.shape[0]
    splitIndex = int(80 / 100 * rowNum)
    train = data.iloc[:splitIndex].reset_index(drop=True) #reset indexes to start from index 0
    test = data.iloc[splitIndex:].reset_index(drop=True)
    return train, test

def eucledianDistance(p1, p2):
    distance = np.sqrt(np.sum((p1 - p2) ** 2))
    return distance



df = pd.read_csv('wine.csv').sample(frac=1)

classNum = len(set(df.iloc[:, 0]))
trainData, testData = trainTestSplit(df)

featuresTrain = trainData.iloc[:, 1:].values
labelsTrain = trainData.iloc[:, 0].values

trainNum = len(featuresTrain)
kk = sqrt(trainNum)


featuresTest = testData.iloc[:, 1:].values
labelsTest = testData.iloc[:, 0].values

# print(labels)

def KNNPredict(featuresTrain, labelsTrain, featureTest, k=15):
    distances = []

    #for every example in the training set, calculate eucledien distance against the test example
    for i, point in enumerate(featuresTrain):
        distances.append((i, eucledianDistance(featureTest, point)))
    distances.sort(key = lambda x : x[1])
    
    labels = []
    for i, distance in distances[:k]:
        labels.append(labelsTrain[i])

    count = Counter(labels)
    label = count.most_common()[0][0]

    return label

def confusionMatrix(predictions, labels, classNum):
    mat = np.zeros((classNum, classNum), dtype=np.int32)

    for i in range(len(predictions)):
        mat[predictions[i] - 1, labels[i] - 1] += 1
    
    return mat

predictions = []
for point in featuresTest:
    predictions.append(KNNPredict(featuresTrain, labelsTrain, point))
accuracy = np.sum(predictions == labelsTest) / len(labelsTest) * 100
print(accuracy)
print(confusionMatrix(predictions, labelsTest, classNum))