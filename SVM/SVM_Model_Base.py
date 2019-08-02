import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# use seaborn plotting default
import seaborn as sns
sns.set()

# create data numpy
from sklearn.datasets.samples_generator import  make_blobs
X, y = make_blobs(n_samples = 50, centers = 2, random_state = 0, cluster_std = 0.6)
plt.scatter(X[:, 0], X[:, 1], s = 50, c = y, cmap = 'autumn')
plt.show()

# assume lots of method to classify
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], s = 50, c = y, cmap = 'autumn')
for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    plt.plot(xfit, m * xfit + b, '-k')
plt.xlim(-1, 3.5)
plt.show()

# assume that every split line has width
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], s = 50, c = y, cmap = 'autumn')
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor = 'none', color = '#AAAAAA', alpha = 0.4)
plt.xlim(-1, 3.5)
plt.show()

# train svm
from sklearn.svm import SVC
model = SVC(kernel = 'linear', C = 1E10)
model.fit(X, y)

# create a complete split line
def plot_svc_decision_function(model, ax = None, plot_support = True):
    """
    plot a decision function of the model
    :param model:
    :param ax:
    :param plot_support:
    :return:
    """
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors = 'k', levels = [-1, 0, 1], alpha = 0.5, linestyles = ['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s = 300, linewidth = 1, facecolors = 'none')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

plt.scatter(X[:, 0], X[:, 1], s = 50, c = y, cmap = 'autumn')
plot_svc_decision_function(model)
plt.show()

print(model.support_vectors_)

# not sensitive to the data
def plot_svm(N = 10, ax = None):
    X, y = make_blobs(n_samples = 200, centers = 2, random_state = 0, cluster_std = 0.6)
    X = X[ : N]
    y = y[ : N]
    model = SVC(kernel = 'linear', C = 1E10)
    model.fit(X, y)

    ax = ax or plt.gca()
    ax.scatter(X[:, 0], X[:, 1], s = 50, c = y, cmap = 'autumn')
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 6)
    plot_svc_decision_function(model, ax)

fig, ax = plt.subplots(1, 2, figsize = (16, 6))
fig.subplots_adjust(left = 0.0625, right = 0.95, wspace = 0.1)
for axi, N in zip(ax, [60, 120]):
    plot_svm(N, axi)
    axi.set_title('N = {0}'.format(N))

plt.show()

print("SVM(support vector machine")