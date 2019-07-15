import pandas as pd

# read csv file
df = pd.read_csv("spam.csv", encoding = 'latin')
print(df.head())

# rename the columns
df.rename(columns = {'v1' : 'Label', 'v2' : 'Text'}, inplace = True)
print(df.head())
# change the label to number
df['numLabel'] = df['Label'].map({'ham' : 0, 'spam' : 1})
# satisfy the number of ham or spam
print("number of ham: ", len(df[df.numLabel == 0]), "number of spam: ", len(df[df.numLabel == 1]))
print("number of total samples: ", len(df))

# statistics text number
text_lengths = [len(df.loc[i, 'Text']) for i in range(len(df))]
print("the minimum length is: ", min(text_lengths))

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

plt.hist(text_lengths, 100, facecolor = 'blue', alpha = 0.5)
plt.xlim([0, 200])
plt.show()

# import English stopwords dictionary
#@from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
#stopset = set(stopwords.words("english"))

# stop words is he she it that... express useless word

# create text vector(base on frequency of the word)
#vectorizer = CountVectorizer(stop_words = stopset, binary=True)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df.Text)
y = df.numLabel
# X is a sparse matrix

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)
print("train number of the X_train ", X_train.shape[0], "test number of the X_test", X_test.shape[0])

# use naive bayes train the model
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

clf = MultinomialNB(alpha = 1.0, fit_prior= True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("accuracy on test data: ", accuracy_score(y_test, y_pred))

# a very important matrix to view the prediction result accuracy
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred, labels = [0, 1]))
# the diagonal express the data of right result of the feature

print("Mail Classification")
