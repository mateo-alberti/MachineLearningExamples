import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Relación no lineal
X = np.array([50, 60, 70, 80, 90, 100, 110, 120]).reshape(-1, 1)
y = np.array([150, 180, 220, 270, 330, 400, 480, 570])  # crecimiento no lineal

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

x_test = np.array([[115]])
x_test_poly = poly.transform(x_test)
pred = model.predict(x_test_poly)
print(f"Precio estimado para 115 m²: ${pred[0]:.2f}K")


# Visualización
X_range = np.linspace(50, 125, 100).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
y_pred = model.predict(X_range_poly)

plt.scatter(X, y, color='blue', label='Datos reales')
plt.plot(X_range, y_pred, color='red', label='Regresión polinómica')
plt.xlabel("Tamaño (m²)")
plt.ylabel("Precio (mil USD)")
plt.legend()
plt.show()