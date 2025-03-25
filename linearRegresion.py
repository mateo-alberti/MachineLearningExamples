import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 🏠 Tamaño de casas (m²)
X = np.array([50, 60, 70, 80, 90, 100, 120, 150]).reshape(-1, 1)
# 💰 Precio (en miles de USD)
y = np.array([150, 180, 200, 220, 250, 270, 310, 360])

# Entrenamiento
model = LinearRegression()
model.fit(X, y)

# Predicción
pred = model.predict([[110]])
print(f"Precio estimado para 110 m²: ${pred[0]:.2f}K")

# 📉 Visualización
plt.scatter(X, y, color='blue', label='Datos reales')
plt.plot(X, model.predict(X), color='red', label='Línea de regresión')
plt.xlabel("Tamaño (m²)")
plt.ylabel("Precio (mil USD)")
plt.legend()
plt.show()