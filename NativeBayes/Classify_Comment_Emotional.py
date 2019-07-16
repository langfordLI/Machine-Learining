from matplotlib import pyplot as plt
import jieba  # word segmentation tools
import regex # regex tools
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def read_data(path, is_pos = None):
    """

    :param path: path to the data
    :param is_pos: data is not a positive samples
    :return:  list of review texts, list of labels
    """
    reviews, labels = [], []
    with open(path, 'r', encoding = 'utf-8') as file:
        review_start = False
        review_text = []
        for line in file:
            line = line.strip()
            if not line: continue
            if not review_start and line.startswith("<review"):
                review_start = True
                if "label" in line:
                    labels.append(int(line.split('"')[-2]))
                    continue
            if review_start and line == "</review>":
                review_start = False
                reviews.append(" ".join(review_text))
                review_text = []
                continue
            if review_start:
                review_text.append(line)
    if is_pos:
        labels = [1] * len(reviews) # print a list of [1, 1, 1, ...]
    elif not is_pos is None:
        labels = [0] * len(reviews)
    return reviews, labels

def process_file():
    """
    :return: read train_number and test_number, then make some process on it
    """
    train_pos_file = "data_sentiment/train.positive.txt"
    train_neg_file = "data_sentiment/train.negative.txt"
    test_comb_file = "data_sentiment/test.combined.txt"

    # rean file, content in the variable
    train_pos_cmts, train_pos_lbs = read_data(train_pos_file, True)
    train_neg_cmts, train_neg_lbs = read_data(train_neg_file, False)
    train_comments = train_pos_cmts + train_neg_cmts
    train_labels = train_pos_lbs + train_neg_lbs
    test_comments, test_labels = read_data(test_comb_file)
    return train_comments, train_labels, test_comments, test_labels

train_comments, train_labels, test_comments, test_labels = process_file()
print(len(train_comments), len(test_comments))
print(train_comments[1], train_labels[1])

def load_stopwords(path):
    """

    :param path: import stopwords from outer file
    :return:
    """
    stopwords = set()
    with open(path, 'r', encoding = 'utf-8') as in_file:
        for linee in in_file:
            stopwords.add(linee.strip())
    return stopwords

def clean_non_chinese_symbols(text):
    """
    deal with not chinese words in the text
    :param text:
    :return:
    """
    text = regex.sub('[!！]+', "!", text)
    text = regex.sub('[?？]+', "?", text)
    text = regex.sub("[a-zA-Z#$%&\'()*+,-./:;：<=>@，。★、…【】《》“”‘’[\\]^_`{|}~]+", " UNK ", text)
    return regex.sub("\s+", " ", text)

def clean_numbers(text):
    """
    deal with the 128 190 NUM
    :param text:
    :return:
    """
    return regex.sub("\d+", ' NUM', text)

def preprocess_text(text, stopwords):
    """
    preprocessing dealing
    :param text:
    :param stopwords:
    :return:
    """
    text = clean_non_chinese_symbols(text)
    text = clean_numbers(text)
    text = " ".join([term for term in jieba.cut(text) if term and not term in stopwords])
    return text

path_stopwords = "./data_sentiment/stopwords.txt"
stopwords = load_stopwords(path_stopwords)

# for train_comments, test_comment deal with the other character
train_comments_new = [preprocess_text(comment, stopwords) for comment in train_comments]
test_comments_new = [preprocess_text(comment, stopwords) for comment in test_comments]

print(train_comments_new[0], test_comments_new[0])

# tf-idf read the feature from the text
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(train_comments_new)
y_train = train_labels
X_test = tfidf.transform(test_comments_new)
y_test = test_labels

print(np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test))

# use Multinomial Naive Bayes to predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("accuracy on test data(Naive Bayes): ", accuracy_score(y_test, y_pred))

# usee KNN to predict
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("accuracy on test data(KNN): ", accuracy_score(y_test, y_pred))

# # use KNN after standard scaler（memory is to small to excute)
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsRegressor
#
# normalizer = StandardScaler() # data is no longer sparse
# X_train_normalized = normalizer.fit_transform(X_train.toarray())
# X_test_normalized = normalizer.transform(X_test.toarray())
#
# knn = KNeighborsRegressor(n_neighbors = 3)
# knn.fit(X_train_normalized, y_train)
#
# y_pred = knn.predict(X_test_normalized)
# print("accuracy on test data(Normalized KNN): ", accuracy_score(y_test, y_pred))

# use Logistic Regression to predict
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver = 'liblinear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("accuracy on test data(Logistic Regression): ", accuracy_score(y_test, y_pred))
