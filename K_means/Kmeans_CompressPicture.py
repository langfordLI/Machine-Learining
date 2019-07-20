from pylab import imread, imshow, figure, show, subplot
import matplotlib.pyplot as plt
from numpy import reshape, uint8, flipud
from sklearn.cluster import KMeans
from copy import deepcopy

img = imread('sample.jpeg')

pixel = reshape(img, (img.shape[0] * img.shape[1], 3)) # picture can change from 8 * 8 * 3 into 64 * 3
pixel_new = deepcopy(pixel)

print(img.shape)
print(pixel_new.shape)

model = KMeans(n_clusters = 5)
labels = model.fit_predict(pixel)
palette = model.cluster_centers_ # compress the color, chage the matrix of the picture all use center point

for i in range(len(pixel)):
    pixel_new[i, :] = palette[labels[i]] # use label as subscript to chagne every matrix context

imshow(reshape(pixel_new, (img.shape[0], img.shape[1], 3)))
plt.show()
