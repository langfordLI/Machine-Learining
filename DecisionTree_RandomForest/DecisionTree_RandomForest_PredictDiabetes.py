import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

col_name = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv("pima-indians-diabetes.csv", header = None, names = col_name)
print(pima.head())

# select need feature
feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
X = pima[feature_cols]
y = pima.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# create decision_classification
clf = DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))



from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file = dot_data, filled = True, rounded = True, special_characters = True,
                feature_names = feature_cols, class_names = ['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())




# create decision tree restrict leaf node
clf = DecisionTreeClassifier(criterion = 'entropy',
                             max_depth = 3)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file = dot_data, filled = True, rounded=True, special_characters=True, feature_names= feature_cols, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes2.png')
Image(graph.create_png())


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    criterion='entropy',
    n_estimators = 1000,
    max_depth = None,
    min_samples_split=10,
    # min_weight_fraction_leaf=0.02
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

print("use decision tree or random forest predict diabetes")
