import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns

df = pd.read_csv('HR_comma_sep.csv', index_col = None)
print(df.isnull().any())
print(df.head())

# rename the data column
df = df.rename(columns = {'satisfaction_level' : 'satisfaction',
                          'last_evaluation' : 'evaluation',
                          'number_project' : 'projectCount',
                          'average_montly_hours' : 'averageMonthlyHours',
                          'time_spend_company' : 'yearsAtCompany',
                          'Work_accident' : 'workAccident',
                          'promotion_last_5years' : 'promotion',
                          'sales' : 'department',
                          'left' : 'turnover'})
# put the label into first columns
front = df['turnover']
df.drop(labels = ['turnover'], axis = 1, inplace = True)
df.insert(0, 'turnover', front)
print(df.head())

# data analysis
print(df.shape)
print(df.dtypes)

# calculate turnover rate
turnover_rate = df.turnover.value_counts() / len(df)
print(turnover_rate)

# describe the data type of the full view
print(df.describe()) # about 23% people is willing to turnover (1 + 1 + ...) / len 1: turnover
# group by the mean value
turnover_Summary = df.groupby('turnover')
print(turnover_Summary.mean())

# correlation analysis
# which feature influence maximum
# which feature correlation maximum
corr = df.corr()
sns.heatmap(corr, xticklabels = corr.columns.values,
            yticklabels=corr.columns.values)
print(corr)
plt.show()

# compare satisfaction--trunover correlation
emp_population = df['satisfaction'][df['turnover'] == 0].mean()
emp_turnover_satisfaction = df[df['turnover'] == 1]['satisfaction'].mean()
print('not leaving satisfaction: ' + str(emp_population))
print('departure satisfaction: ' + str(emp_turnover_satisfaction))

# t_test compare the satisfaction
import scipy.stats as stats
print(stats.ttest_1samp(a = df[df['turnover'] == 1]['satisfaction'], # departure employee samples # ttest_1sampe 1 is one
                  popmean = emp_population) # not leaving employee satisfaction
      ) # pvalue(0) is small show that the turnover and satisfaction significantly different
degree_freedom = len(df[df['turnover'] == 1])
LQ = stats.t.ppf(0.025, degree_freedom) # 95% left margin of confidence interval
RQ = stats.t.ppf(0.975, degree_freedom) # 95% right margin of confidence interval
print("The t-distribution left margin: " + str(LQ))
print('The t-distribution right margin: ' + str(RQ))
# the probability density of the evaluation
fig = plt.figure(figsize = (15, 4))
ax = sns.kdeplot(df.loc[(df['turnover'] == 0), 'evaluation'], color = 'b', shade = True, label = 'no turnover')
ax = sns.kdeplot(df.loc[(df['turnover'] == 1), 'evaluation'], color = 'r', shade = True, label = 'trunover')
ax.set(xlabel = 'Employee Evaluation', ylabel = 'Frequency')
plt.title('Employee Evaluation Distirbution - Turnover V.S. No Turnover')
plt.show()

# probability density of the averageMonthlyHours-turnover
fig = plt.figure(figsize = (15, 4))
ax = sns.kdeplot(df.loc[(df['turnover'] == 0), 'averageMonthlyHours'], color = 'b', shade = True, label = 'no turnover')
ax = sns.kdeplot(df.loc[(df['turnover'] == 1), 'averageMonthlyHours'], color = 'r', shade = True, label = 'turnover')
ax.set(xlabel = 'Employee Average Monthly Hours', ylabel = 'Frequency')
plt.title('Employee AverageMonthly Hours Distribution - Turnover V.S. No Turnover')
plt.show()

# probability density of the satisfaction
fig = plt.figure(figsize = (15, 8))
ax = sns.kdeplot(df.loc[(df['turnover'] == 0), 'satisfaction'], color = 'b', shade = True, label = 'no turnover')
ax = sns.kdeplot(df.loc[(df['turnover'] == 1), 'satisfaction'], color = 'r', shade = True, label = 'turnover')
plt.title('Employee Satisfaction Distribution - Turnover V.S. No Turnover')
plt.show()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve
# turn the string to integrate
df['department'] = df['department'].astype('category').cat.codes
df['salary'] = df['salary'].astype('category').cat.codes
print(df.head())

target_name = 'turnover'
X = df.drop('turnover', axis = 1)
y = df[target_name]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state=123, stratify=y) # keep the original leaving rate smaples

# create decision_tree & random_forest
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# decision tree
dtree = tree.DecisionTreeClassifier(criterion = 'entropy', # and for 'gini'
                                    # max_depth = 3 # define the tree's depth to prevent overfitting
                                    min_weight_fraction_leaf=0.01 # define leaf node include sample at least to prevent overfitting
 )
dtree = dtree.fit(X_train, y_train)
print("\n\n--decision tree--")
dt_roc_auc = roc_auc_score(y_test, dtree.predict(X_test))
print("decision_tree AUC = %2.2f" %dt_roc_auc)
print(classification_report(y_test, dtree.predict(X_test)))

# random_forest
rf = RandomForestClassifier(
    criterion = 'entropy',
    n_estimators = 1000, # the number of decision_tree
    max_depth = None, # define depth of the tree to prevent overfittin
    min_samples_leaf=10 # define at least samples can bifurcate
    #min_weight_fraction_leaf = 0.02 # define at least leaf node numbers to prevent overfitting
)
rf.fit(X_train, y_train)
print("\n\n--ramdomForest--")
rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test))
print("random forest AUC = %2.2f" %rf_roc_auc)
print(classification_report(y_test, rf.predict(X_test)))

# roc map
from sklearn.metrics import roc_curve
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
dt_fpr, dt_tpr, dt_thresholds = roc_curve(y_test, dtree.predict_proba(X_test)[:, 1])
plt.figure()

# random forest ROC
plt.plot(rf_fpr, rf_tpr, label = 'Random Forest (area = %0.2f)' % rf_roc_auc)
# decision tree ROC
plt.plot(dt_fpr, dt_tpr, label = 'Decision Tree (area = %0.2f)' % dt_roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim(([0.0, 1.05]))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc = 'lower right')
plt.show()

# analysis the importance of the diferent feature
# decision tree
importance = dtree.feature_importances_
feat_names = df.drop(['turnover'], axis = 1).columns

indices = np.argsort(importance)[::-1]
plt.figure(figsize = (12, 6))
plt.title(("Feature importance by DecisionTreeClassifier"))
plt.bar(range(len(indices)), importance[indices], color = 'lightblue', align = 'center')
plt.step(range(len(indices)), np.cumsum(importance[indices]), where = 'mid', label = 'Comulative')
plt.xticks(range(len(indices)), feat_names[indices], rotation = 'vertical', fontsize = 14)
plt.xlim(([-1, len(indices)]))
plt.show()

# random forest
importance = rf.feature_importances_
feat_names = df.drop(['turnover'], axis = 1).columns

indices = np.argsort(importance)[::-1]
plt.figure(figsize = (12, 6))
plt.title(("Feature importance by Random Forest"))
plt.bar(range(len(indices)), importance[indices], color = 'lightblue', align = 'center')
plt.step(range(len(indices)), np.cumsum(importance[indices]), where = 'mid', label = 'Comulative')
plt.xticks(range(len(indices)), feat_names[indices], rotation = 'vertical', fontsize = 14)
plt.xlim(([-1, len(indices)]))
plt.show()


print("DecisionTree_RandomForest")