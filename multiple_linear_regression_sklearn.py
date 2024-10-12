import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


# Input data: Area (m²), Bedrooms, Age (years), Price (thousands of pesos)
data = np.array([
    [50, 1, 5, 200],
    [60, 2, 10, 250],
    [70, 2, 3, 280],
    [80, 3, 8, 300],
    [90, 3, 15, 320],
    [100, 4, 2, 340],
    [110, 4, 1, 360],
    [120, 5, 6, 390],
    [130, 5, 4, 400],
    [140, 6, 20, 420]
])

# Split the data into input features and target variable
X = data[:, :3]  # Area, Bedrooms, Age
Y = data[:, 3]   # Price

# Normalize data
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
print(X_norm.shape)
print(Y.shape)

sgdr = SGDRegressor(max_iter=10_000)
sgdr.fit(X_norm, Y)

w = sgdr.coef_
b = sgdr.intercept_
print(f"w: {w}, b:{b}, Number of iterations: {sgdr.n_iter_}")


# Linear Regresion Predictions
Y_pred = sgdr.predict(X_norm)
print(f"Y_pred: {Y_pred}")

# Prediction for a single house. Area (m²), Bedrooms, Age (years)
X_new_house = np.array([[
    125, 4, 3
]])
X_new_house_norm = scaler.transform(X_new_house)
Y_pred_new_house = sgdr.predict(X_new_house_norm)
price = Y_pred_new_house[0] * 1000
print(f"Predicted price for new house with an Area of {X_new_house[0, 0]}m², {X_new_house[0, 1]} bedrooms and {X_new_house[0, 2]} years is: {price:,.2f}mxn")

#Visualization
fig, axs = plt.subplots(3, 1, figsize=(15, 10), constrained_layout=True)

axs[0].set_title('House Prices Based on Area (m²)')
axs[0].scatter(X[:, 0], Y, color="red", label="Area (m²)", s=75, edgecolor='black')
axs[0].scatter(X_new_house[:, 0], Y_pred_new_house, color="yellow", s=100, label="New House Prediction", edgecolor='black')
axs[0].scatter(X[:, 0], Y_pred, color="blue", label="Regression Prediction", s=75, edgecolor='black')
axs[0].set_xlabel("Area (m²)")
axs[0].set_ylabel("House Price")
axs[0].grid(linestyle="--", color="green")
axs[0].legend()

axs[1].set_title('House Prices Based on Bedrooms')
axs[1].scatter(X[:, 1], Y, color="red", label="Bedrooms (scaled)", s=75, edgecolor='black')
axs[1].scatter(X_new_house[:, 1], Y_pred_new_house, color="yellow", s=100, label="New House Prediction", edgecolor='black')
axs[1].scatter(X[:, 1], Y_pred, color="blue", label="Regression Prediction", s=75, edgecolor='black')
axs[1].set_xlabel("Bedrooms")
axs[1].set_ylabel("House Price")
axs[1].grid(linestyle="--", color="green")
axs[1].legend()

axs[2].set_title('House Prices Based on Age (years)')
axs[2].scatter(X[:, 2], Y, color="red", label="Age (scaled)", s=75, edgecolor='black')
axs[2].scatter(X_new_house[:, 2], Y_pred_new_house, color="yellow", s=100, label="New House Prediction", edgecolor='black')
axs[2].scatter(X[:, 2], Y_pred, color="blue", label="Regression Prediction", s=75, edgecolor='black')
axs[2].set_xlabel("Age (years)")
axs[2].set_ylabel("House Price")
axs[2].grid(linestyle="--", color="green")
axs[2].legend()
plt.show()