from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap = 'bone')
    axi.set(xticks = [], yticks = [], xlabel = faces.target_names[faces.target[i]])
plt.show()
# use pca analysis the main composition
"""
plat the image into a one-dimensional vector, then use PCA to get the main information
"""
from sklearn.svm import SVC
from sklearn.decomposition import  PCA
from sklearn.pipeline import make_pipeline
pca = PCA(n_components=150, whiten = True, random_state=42)
svc = SVC(kernel='linear', class_weight='balanced')
model = make_pipeline(pca, svc)

# train the data
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target,
                                                random_state=42)
# use cross validation to ensure parameter
from sklearn.model_selection import GridSearchCV

param_grid = {'svc__C': [1, 5, 10, 50]}
grid = GridSearchCV(model, param_grid)

grid.fit(Xtrain, ytrain)
print(grid.best_params_)

model = grid.best_estimator_
yfit = model.predict(Xtest)

fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62, 47), cmap = 'bone')
    axi.set(xticks = [], yticks = [])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],color = 'blace' if yfit[i] == ytest[i] else 'red')


fig.suptitle('Predicted Names; Incorrect Labels in Red', size = 14)
plt.show()

from sklearn.metrics import classification_report
print(classification_report(ytest, yfit, target_names = faces.target_names))

from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set()
mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T, square = True, annot = True, fmt = 'd', cbar = False,
            xticklabels = faces.target_names,
            yticklabels = faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
print("SVC_Face_Recognization")

