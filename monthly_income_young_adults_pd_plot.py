'''
Problem:
We want to predict the monthly income of young adults in Mexico based on the following factors:

Years of Education: The number of years the person has studied.
Weekly Working Hours: The number of hours worked per week.
Age: The age of the person.
English Proficiency: A scale from 1 to 5 (1 = none, 5 = fluent).
Residence Area: Coded as 1 for urban areas and 0 for rural areas.

The goal is to create a linear regression model that predicts the monthly income in Mexican pesos (MXN) based on these factors.
'''
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


training_data = pd.read_csv('training_data.csv')
# training_data = pd.read_csv('training_data_large.csv')

X = training_data.iloc[:, :-1].values
Y = training_data.iloc[:, -1:].values.ravel()

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

sgdr = SGDRegressor(max_iter=100_000)
sgdr.fit(X_norm, Y)
Y_pred = sgdr.predict(X_norm)

w = sgdr.coef_
b = sgdr.intercept_

new_young_adults_data = pd.DataFrame({
    "Years of Education": [12, 14, 10, 16, 18, 11],
    "Weekly Working Hours": [40, 38, 30, 45, 50, 36],
    "Age": [26, 29, 22, 32, 41, 25],
    "English Proficiency": [2, 3, 1, 4, 5, 2],
    "Residence Area": [1, 0, 1, 1, 1, 0]
})

X_new_young_adults_data = new_young_adults_data.iloc[:, :].values
X_new_young_adults_data_norm = scaler.transform(X_new_young_adults_data)
Y_new_young_adults_data_predict = sgdr.predict(X_new_young_adults_data_norm)


fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 10), constrained_layout=True)
axs = axs.flatten()

for index, column in enumerate(training_data.columns[:-1]):
    X_title = column
    X_index = training_data[column].values

    sorted_indexes = np.argsort(X_index)
    X_index_sorted = X_index[sorted_indexes]
    Y_sorted = Y[sorted_indexes]
    Y_pred_sorted = Y_pred[sorted_indexes]

    axs[index].scatter(X_index_sorted, Y_sorted, label="Actual Income", s=70, color="orangered", edgecolor="black", alpha=0.5)
    axs[index].scatter(X_index_sorted, Y_pred_sorted, color="blue", s=70, label="Prediction", edgecolor="black", alpha=0.5)
    axs[index].scatter(X_new_young_adults_data[:, index], Y_new_young_adults_data_predict, color="yellow", s=70, label="Young Adult Prediction", edgecolor="black")

    axs[index].set_ylabel("Monthly Income (MXN)")
    axs[index].set_xlabel(X_title)
    axs[index].grid(linestyle="--", color="green")
    axs[index].legend()

axs[-1].axis('off')
plt.show()
