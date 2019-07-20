from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('data.csv')
print("Input Data and Shape")
print(data.shape)
print(data.head())

# extract the data and visualization
f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))
#X = np.random.random((200, 2)) * 10
plt.scatter(X[:, 0], X[:, 1], c = 'black', s = 8) # s show the point size
plt.show()

# train K-means
k = 3
C_x = np.random.randint(0, np.max(X), size = k)
C_y = np.random.randint(0, np.max(X), size = k)
C = np.array(list(zip(C_x, C_y)), dtype = np.float32)
print("initialization center point: ")
print(C)

plt.scatter(X[:, 0], X[:, 1], c = '#050505', s = 7)
plt.scatter(C[:, 0], C[:, 1], marker = '*', s = 300, c = 'g')
plt.show()

C_old = np.zeros(C.shape) # for initialization [0, 0, 0]
clusters = np.zeros(len(X)) # initialize the data for length[0, 0, 0, 0]
#print(C_old)
#print(clusters)

def dist(a, b, ax = 1):
    return np.linalg.norm(a - b, axis = ax)

error = dist(C, C_old, None)
while error != 0:
    for i in range(len(X)):
        distances = dist(X[i], C) # get the every point-C distance
        cluster = np.argmin(distances) # get the minimum distance, there is 3 points. get the subscript of the point
        clusters[i] = cluster
        # for one cycle you can get one point label for which point
    C_old = deepcopy(C) # save the old center point to calculate the distance
    # calculate the new center point
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i] # get every label X point
        C[i] = np.mean(points, axis = 0)
    error = dist(C, C_old, None)

colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
    points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
    ax.scatter(points[:, 0], points[:, 1], s = 7, c = colors[i])
ax.scatter(C[:, 0], C[:, 1], marker = '*', s = 200, c = '#050505')
plt.show()


# create random number
X = np.random.random((200, 2)) * 10
plt.scatter(X[:, 0], X[:, 1], c = 'black', s = 8) # s show the point size
plt.show()

# train K-means
k = 3
C_x = np.random.randint(0, np.max(X), size = k)
C_y = np.random.randint(0, np.max(X), size = k)
C = np.array(list(zip(C_x, C_y)), dtype = np.float32)
print("initialization center point: ")
print(C)

plt.scatter(X[:, 0], X[:, 1], c = '#050505', s = 7)
plt.scatter(C[:, 0], C[:, 1], marker = '*', s = 300, c = 'g')
plt.show()

C_old = np.zeros(C.shape) # for initialization [0, 0, 0]
clusters = np.zeros(len(X)) # initialize the data for length[0, 0, 0, 0]
#print(C_old)
#print(clusters)

def dist(a, b, ax = 1):
    return np.linalg.norm(a - b, axis = ax)

error = dist(C, C_old, None)
while error != 0:
    for i in range(len(X)):
        distances = dist(X[i], C) # get the every point-C distance
        cluster = np.argmin(distances) # get the minimum distance, there is 3 points. get the subscript of the point
        clusters[i] = cluster
        # for one cycle you can get one point label for which point
    C_old = deepcopy(C) # save the old center point to calculate the distance
    # calculate the new center point
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i] # get every label X point
        C[i] = np.mean(points, axis = 0)
    error = dist(C, C_old, None)

colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
    points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
    ax.scatter(points[:, 0], points[:, 1], s = 7, c = colors[i])
ax.scatter(C[:, 0], C[:, 1], marker = '*', s = 200, c = '#050505')
plt.show()

