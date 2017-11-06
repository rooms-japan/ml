# Linear regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error, r2_score

# Import data and normalize it
normalize = lambda x : (x.values - np.mean(x.values) ) / np.std(x.values) 
data = pd.read_csv("data.txt", sep="\t")
ndata = pd.DataFrame()

mean_rent = np.mean(data["rent"])
std_rent = np.std(data["rent"])

# Remove ward (it is categorical)
for col in data.columns:
	if not "ward" in col:
		kwargs = { col: normalize(data[col])}
		ndata = ndata.assign(**kwargs)
# print(ndata.head())

# Fit linear regression, remove rent since target variable
X = ndata[ndata.columns.difference(["ward", "rent"])]
X_train = X[:-200]
X_test = X[-200:]

y = ndata["rent"]
y_train = y[:-200]
y_test = y[-200:]
print(X_test.shape, y_test.shape)

linmodel = lm.LinearRegression(fit_intercept=False)
linmodel.fit(X_train, y_train)
y_pred = linmodel.predict(X_test)

print('Coefficients: \n', linmodel.coef_)
# print('Feature importance: \n', linmodel._feature)
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))
# print('Intercept score: %.2f' % linmodel.intercept_)

# Plot
plt.xlabel("Real rent")
plt.ylabel("Predicted rent")
plt.scatter(data["rent"][-200:], (y_pred * std_rent) + mean_rent, color='black')
# plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
# Plot residuals
# plt.scatter(linmodel.predict(X_train), linmodel.predict(X_train) - y_train, c='b', s=40, alpha=0.5)
# plt.scatter(linmodel.predict(X_test), linmodel.predict(X_test) - y_test, c='g', s=40)
# plt.hlines(y=0, xmin=0, xmax=10)

plt.show()