import numpy as np
import pandas as pd
from metric_learn import LMNN, NCA
from collections import Counter
import matplotlib.pyplot as plt

def trainTestSplit(features, labels):
    rowNum = features.shape[0]
    splitIndex = int(80 / 100 * rowNum)
    featuresTrain = features[:splitIndex] #We drop the index respectively relabel the index
    labelsTrain = labels[:splitIndex] #We drop the index respectively relabel the index
    #starting form 0, because we do not want to run into errors regarding the row labels / indexes
    featuresTest = features[splitIndex:]
    labelsTest = labels[splitIndex:]
    return featuresTrain, labelsTrain, featuresTest, labelsTest

def eucledianDistance(p1, p2):
    distance = np.sqrt(np.sum((p1 - p2) ** 2))
    return distance


df = pd.read_csv('wine.csv').sample(frac=1)

features = df.iloc[:, 1:].values
labels = df.iloc[:, 0].values

#LMNN
lmnn = LMNN(k=5, learn_rate=1e-6)
features1 = lmnn.fit_transform(features, labels)

#NCA
nca = NCA(max_iter=1000)
features2 = nca.fit_transform(features, labels)

classNum = len(set(df.iloc[:, 0]))
featuresTrain1, labelsTrain1, featuresTest1, labelsTest1 = trainTestSplit(features1, labels)

featuresTrain2, labelsTrain2, featuresTest2, labelsTest2 = trainTestSplit(features2, labels)
# print(labels)

def KNNPredict(featuresTrain, labelsTrain, featureTest, k):
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
#1
errorRate1 = np.zeros(20)
accuracy1 = np.zeros(20)
for k in range(20):
    predictions1 = []
    for point in featuresTest1:
        predictions1.append(KNNPredict(featuresTrain1, labelsTrain1, point, 2*k+1))
    accuracy1[k] = np.sum(predictions1 == labelsTest1) / len(labelsTest1) * 100
    errorRate1[k] = (np.mean(predictions1 != labelsTest1))
    if k == 2:
        print("LMNN accuracy: ",accuracy1[2])

#2
errorRate2 = np.zeros(20)
accuracy2 = np.zeros(20)
for k in range(20):
    predictions2 = []
    for point in featuresTest2:
        predictions2.append(KNNPredict(featuresTrain2, labelsTrain2, point, 2*k+1))
    accuracy2[k] = np.sum(predictions2 == labelsTest2) / len(labelsTest2) * 100
    errorRate2[k] = (np.mean(predictions2 != labelsTest2))
    if k == 2:
        print("NCA accuracy: ",accuracy2[2])


k = np.arange(1,41,2)

print("Minimum error1:-",min(errorRate1),"at K =",2*np.argmin(errorRate1)+1)
print("Minimum error2:-",min(errorRate2),"at K =",2*np.argmin(errorRate2)+1)

#plot
plt.plot(k,errorRate1,color='blue', linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.plot(k,errorRate2,color='blue', linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.show()

# plt.plot(k,accuracy1, marker="o",label="LMNN")
# plt.plot(k,accuracy2,marker="o",label="NCA")


# plt.xlabel("K")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.grid()
# plt.show()