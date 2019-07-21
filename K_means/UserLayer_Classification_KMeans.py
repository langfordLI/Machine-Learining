import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns

data_offer = pd.read_excel('WineKMC.xlsx', sheetname = 0)
data_offer.columns = ["offer_id", "campaign", "varietal", "min_qty", "discount", "origin", "past_peak"]
print(data_offer.head())

data_transactions = pd.read_excel("WineKMC.xlsx", sheetname = 1)
data_transactions.columns = ['customer_name', 'offer_id']
data_transactions['n'] = 1
print(data_transactions.head())

# merge the two excel sheet
cust_compare = data_transactions.merge(data_offer, on = 'offer_id')
cust_compare = cust_compare.drop(['campaign', 'varietal', 'min_qty', 'discount', 'origin', 'past_peak'], axis = 1)
table = pd.pivot_table(cust_compare, index = 'customer_name', columns = 'offer_id', aggfunc = np.sum, fill_value=0)
print(table)

# select K number in k-means
"""
compare the inertia of the different K value
"""
# offers = table.columns.get_level_value('offer_id')
# x_cols = np.matrix(offers)
SS = []
from sklearn.cluster import KMeans
for K in range(2, 20):
    kmeans = KMeans(n_clusters = K).fit(table) # using all default values from method
    SS.append(kmeans.inertia_)

plt.plot(range(2, 20), SS)
plt.xlabel('K')
plt.ylabel('SS')
plt.show()

# choosing k = 5
kmeans_5 = KMeans(n_clusters = 5).fit_predict(table)
points = list((kmeans_5))
d = {x : points.count(x) for x in points}
heights = list(d.values())
plt.bar(range(5), heights)
plt.xlabel('Cluster')
plt.ylabel('number of samples')
plt.show()
# different samples lead to different result


# dimension reduction to visualization
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
data_new = pca.fit_transform(table)
print(table.shape)
print(data_new.shape)

colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(5):
    points = np.array([data_new[j] for j in range(len(data_new)) if kmeans_5[j] == i])
    ax.scatter(points[:, 0], points[:, 1], s = 7, c = colors[i])

plt.show()
