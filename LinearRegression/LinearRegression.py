import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("exercise\height.vs.temperature.csv")

plt.figure(figsize = (16, 8))
plt.scatter(data['height'], data['temperature'], c = 'black')
plt.xlabel("height")
plt.ylabel("temperature")
plt.show()

# turn to the numpy matrix
X = data['height'].values.reshape(-1, 1)
y = data['temperature'].values.reshape(-1, 1)

# use library functions to get the linear regression
reg = LinearRegression()
reg.fit(X, y)


# draw the chart
print('a = {:.5}'.format(reg.coef_[0][0]))
print('b = {:.5}'.format(reg.intercept_[0]))

# draw thee chart and line in one picture
predictions = reg.predict(X)
plt.figure(figsize = (16, 8))
plt.scatter(data['height'], data['temperature'], c = 'black')
plt.plot(data['height'], predictions, c = 'blue', linewidth = 2)
plt.xlabel("height")
plt.ylabel("tmeperature")
plt.show()

predictions = reg.predict([[8000]])
print("the predictions of the 8000 height {:.5}".format(predictions[0][0]))
