# Compare decision tree regressor with any other regressor

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# generate data
X = np.linspace(0, 4)
noise = 0.35 * np.random.rand(len(X))
y = np.add(np.sin(X), noise)
# reshape because data has a single feature
X_reshaped = X.reshape(-1, 1)  # [0,1,2] -> [[0], [1], [2]]

# create decision tree regressor
dt = DecisionTreeRegressor(max_depth=3)
dt.fit(X_reshaped, y)
dt_y = dt.predict(X_reshaped)

# create linear regressor
pf = PolynomialFeatures(degree=2)
X_poly = pf.fit_transform(X_reshaped)

lr = LinearRegression()
lr.fit(X_poly, y)
lr_y = lr.predict(X_poly)

# plot chart
plt.title("Regression")
plt.scatter(X, y, s=20, color="silver")
plt.plot(X, dt_y, label="Decision Tree", color="darkcyan")
plt.plot(X, lr_y, label="Linear Regression + PF", color="darkorange")
plt.legend()
plt.savefig("output/01.png")
plt.show()
