from sklearn import datasets # some library function
from collections import Counter # for vote the number
from sklearn.model_selection import train_test_split
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2003)

def nuc_dis(instance1, instance2):
    """calculate Euclidean  distance"""
    return np.sqrt(sum((instance1 - instance2)**2))

def knn_classify(X, y, testInstance, k):
    """
    X: train_number feature
    Y: train_number label
    testInstance: # of samples
    k: the number of the closest
    """
    distances = [nuc_dis(x, testInstance) for x in X] # list type array
    kneighbers = np.argsort(distances)[:k] # sort from low to high
    count = Counter(y[kneighbers])
    return count.most_common()[0][0]

predictions = [knn_classify(X_train, y_train, data, 3) for data in X_test]
correct = np.count_nonzero((predictions == y_test) == True)
print("Accuracy = %.3f" %(correct / len(y_test)))
