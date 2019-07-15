import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size = 14)
import seaborn as sns
sns.set(style = "white") # set the background of the seaborn
sns.set(style = "whitegrid", color_codes = True)

# import the data
df = pd.read_csv("titanic_data.csv")
print(df.head())

# view missing data
print(df.isnull().sum())
print("*********************")

data = df.copy()
# age missing
print("age percentage of the missing data %.2f%%" %((df['age'].isnull().sum() / df.shape[0]) * 100))
# 20.15% age missing, search for the distribution of the age
ax = df['age'].hist(bins = 15, color = 'teal', alpha = 0.6)
ax.set(xlabel = 'age')
plt.xlim(-10, 85) # set the x-coordinate range
plt.show()
# use median number replace the missing age
data["age"].fillna(df["age"].median(skipna = True), inplace = True)

# boarding place missing
print("boarding place percentage of the missing data %.2f%%" %((df['embarked'].isnull().sum() / df.shape[0]) * 100))
# % age missing, search for the distribution of the boarding place
print('group by the boarding place (C = Cherbourg, Q = Queenstown, S = Southampton')
sns.countplot(x = 'embarked', data =df, palette = 'Set2')
plt.show()
# use the number of most replace the missing age
data["embarked"].fillna(df['embarked'].value_counts().idxmax(), inplace = True)

# cabin missing
print("cabin percentage of the missing data %.2f%%" %((df['cabin'].isnull().sum() / df.shape[0]) * 100))
# % age missing, search for the distribution of the boarding place
# cabin number miss a lot so delete the cabin number
data.drop('cabin', axis = 1, inplace = True)


#np.isnan(data).any()
#print(data.drop(['pclass', 'survived', 'name', 'sex', 'sibsp', 'parch', 'parch', 'ticket', 'fare'], axis = 1, inplace = True))
data.dropna(axis = 0, how = 'any', inplace = True)
print(data.isnull().sum())

# view the age feature distribution before the dispose and after
plt.figure(figsize = (15, 8))
ax = df["age"].hist(bins = 15, normed = True, stacked = True, color = 'teal', alpha = 0.6)
df["age"].plot(kind = 'density', color = 'teal')
ax = data["age"].hist(bins = 15, normed = True, stacked = True, color = 'orange', alpha = 0.5)
data["age"].plot(kind = 'density', color = 'orange')
ax.legend(['Raw Age', 'Adjusted Age'])
ax.set(xlabel = 'Age')
plt.xlim(-10, 85)
plt.show()


# multicolinearity problem (variablees are highly correlated)
# delete the sibsp and parch create a new line to achieve record the TraveAlone
data['TravelAlone'] = np.where((data["sibsp"] + data["parch"]) > 0, 0, 1)
data.drop('sibsp', axis = 1, inplace = True)
data.drop('parch', axis = 1, inplace = True)

# code the embarkeed and sex in One-hot coding, deprecate the name and sex
final = pd.get_dummies(data, columns = ["embarked", "sex"])
final.drop('name', axis = 1, inplace = True)
final.drop('ticket', axis = 1, inplace = True)
print(final.head())

"""continue analysis the data"""

# view the age respectively in the survived people and not survived people
plt.figure(figsize = (15, 8))
ax = sns.kdeplot(final["age"][final.survived == 1], color = "darkturquoise", shade = True)
sns.kdeplot(final["age"][final.survived == 0], color = "lightcoral", shade = True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Age for Surviving Population and Deceased Population')
ax.set(xlabel = 'Age')
plt.xlim(-10, 85)
plt.show()

# view the ticket price relationship of the survivable
plt.figure(figsize = (15, 8))
ax = sns.kdeplot(final["fare"][final.survived == 1], color = "darkturquoise", shade = True)
sns.kdeplot(final["fare"][final.survived == 0], color = "lightcoral", shade = True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Fare for Surviving ')
ax.set(xlabel = 'Fare')
plt.xlim(-20, 200)
plt.show()

# search the cabin relationship of the survivable
sns.barplot('pclass', 'survived', data = df, color = "darkturquoise")
plt.show()

# view the boarding place rate
sns.barplot('embarked', 'survived', data = df, color = "teal")
plt.show()

# view the Travel alone rate
sns.barplot('TravelAlone', 'survived', data = final, color = "mediumturquoise")
plt.show()

# view thee sex rate
sns.barplot('sex', 'survived', data = df, color = "aquamarine")
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""train Logistic Regression to predict"""
# use following feature to predict
cols = ["age", "fare", "TravelAlone", "pclass", "embarked_C", "embarked_S", "sex_male"]
X = final[cols] # select X except for the below feature
y = final['survived']

# part the X, y into train and test part
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Train / test split results %2.3f' % accuracy_score(y_test, y_pred))

print("titanic_logistic_regression")
