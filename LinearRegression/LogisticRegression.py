import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc("font", size = 14) # set the font and size
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style = "white")
sns.set(style = "whitegrid", color_codes = True) # set background

data = pd.read_csv("banking.csv", header = 0)
data = data.dropna()  # delete the
print(data.shape)
print(list(data.columns))

print(data['education'].unique()) # show the education categorical
data['education'] = np.where(data['education'] == 'basic.9y', 'Basic', data['education'])
data['education'] = np.where(data['education'] == 'basic.6y', 'Basic', data['education'])
data['education'] = np.where(data['education'] == 'basic.4y', 'Basic', data['education'])

print(data['education'].unique())

data['y'].value_counts() #print the y prediction value

"""sns.countplot(x = 'y', data = data, palette = 'hls')
plt.show()
plt.savefig('count_plot')

count_no_sub = len(data[data['y'] == 0]) # count y = 0
count_sub = len(data[data['y'] == 1]) # count y = 1
pct_of_no_sub = count_no_sub / (count_no_sub + count_sub)
print('not to open an account: %.2f%%' % (pct_of_no_sub * 100))
pct_of_sub = count_sub / (count_no_sub + count_sub)
print('open an account: %.2f%%' % (pct_of_sub * 100))"""

print(data.groupby('y').mean()) # get the average of the y = 0 or y = 1 relative variable value's mean

"""calculate distribution of eigenvalues"""
print(data.groupby('job').mean())
print(data.groupby('marital').mean())
print(data.groupby('education').mean())

# """draw the plot in box"""
# table = pd.crosstab(data.job, data.y)
# table.div(table.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True)
# plt.title('Stacked Bar Chart of Job title vs Purchase')
# plt.xlabel('job')
# plt.ylabel('Proportion of Purchase')
# plt.savefig('purchase_vs_job')
# plt.show() # from the chart we can get the job is an important element of the willing to open an account
#
# """same to draw the marital box analysis"""
# table = pd.crosstab(data.marital, data.y)
# table.div(table.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True)
# plt.title('Stacked Bar Chart of Marital Status vs Purchase')
# plt.xlabel('Marital Status')
# plt.ylabel('Proportion of Customeers')
# plt.savefig('maarital_vs_pur_stack')
# plt.show()# data plot is an important element of the chart
# #from the chart marital or not is not relative to the account
#
# table = pd.crosstab(data.education, data.y)
# table.div(table.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True)
# plt.title('Stack Bar Chart of Education vs Purchase')
# plt.xlabel('Education')
# plt.ylabel('Proportion of Customers')
# plt.savefig('edu_vs_pur_stack')
# plt.show() # maybe education can be a significant element
#
# table = pd.crosstab(data.day_of_week, data.y)
# table.div(table.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True)
# plt.title('Stacked Bar Chart of Day of Week vs Purchase')
# plt.xlabel('Day of Week')
# plt.ylabel('Proportion of Purchase')
# plt.savefig('dow_vs_purchase')
# plt.show() # work time every week maybe not an element because of the percentage

# according to the analysis to decide the element selection
cat_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
for var in cat_vars:
    cat_list = pd.get_dummies(data[var], prefix = var) # pd.get_dummies one-hot code
    data = data.join(cat_list)

data_final = data.drop(cat_vars, axis = 1)
data_final.columns.values

# use SMOTE to exceed sample
X = data_final.loc[:, data_final.columns != 'y']
y = data_final.loc[:, data_final.columns == 'y'].values.ravel()

from imblearn.over_sampling import SMOTE
os = SMOTE(random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=  0)
columns = X_train.columns
os_data_X, os_data_y = os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data = os_data_X, columns = columns)
os_data_y = pd.DataFrame(data = os_data_y, columns = ['y'])
# check the numbers of our data
print("exceed sample count: ", len(os_data_X))
print("not account user count: ", len(os_data_y[os_data_y['y'] == 0]))
print("account user count: ", len(os_data_y[os_data_y['y'] == 1]))
print("percentage of the not account count: ", len(os_data_y[os_data_y['y'] == 0]) / len(os_data_X))
print("percentage of the account count: ", len(os_data_y[os_data_y['y'] == 1]) / len(os_data_X))
# get the prefect balanced number

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(os_data_X, os_data_y.values.reshape(-1))

# calculate the accuracy of the prediction
y_pred = logreg.predict(X_test)
print('test the accuracy on the test_data: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# visualization
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label = 'Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc = 'lower right')
plt.savefig('Log_ROC')
plt.show()


print("Logistic Regression")
