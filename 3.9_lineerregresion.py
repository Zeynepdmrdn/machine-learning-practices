
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt





# veri olustur
X = np.random.rand(100,1)
y = 3 + 4 * X # y = 3 + 4x

# plt.scatter(X,y)


lin_reg = LinearRegression()
lin_reg.fit(X,y)


plt.figure()
plt.scatter(X,y)
plt.plot(X, lin_reg.predict(X), color = "red")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regresyon")
plt.show()
