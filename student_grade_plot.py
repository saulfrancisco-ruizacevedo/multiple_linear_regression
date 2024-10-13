'''
Regression model to predict a student's final grade based on the five attributes.

Description of the Parameters:

Study Hours: The number of hours a student studies each week.
Attendance (%): The percentage of classes attended by the student.
Extracurricular Activities: The count of extracurricular activities the student participates in.
Parental Involvement: A scale from 1 to 5 indicating the level of parental involvement in the student's education (1 = low, 5 = high).
Age: The age of the student.
Final Grade (%): The final grade percentage achieved by the student.
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

data = np.array([
    ["Study Hours", "Attendance (%)", "Extracurricular Activities", "Parental Involvement", "Age", "Final Grade (%)"],
    [5, 90, 2, 4, 16, 85],
    [3, 80, 1, 3, 17, 75],
    [4, 95, 3, 5, 15, 90],
    [2, 70, 0, 2, 16, 70],
    [6, 85, 1, 4, 17, 88],
    [1, 60, 0, 1, 15, 65],
    [7, 100, 4, 5, 18, 92],
    [8, 95, 5, 5, 18, 94],
    [4, 80, 2, 3, 17, 76],
    [5, 90, 2, 4, 16, 83]
])


X = data[1:, :-1].astype(float)
Y = data[1:, -1].astype(float)

# Normalize data
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

sgdr = SGDRegressor(max_iter=100_000)
sgdr.fit(X_norm, Y)

w = sgdr.coef_
b = sgdr.intercept_

Y_pred = sgdr.predict(X_norm)

# Study Hours, Attendance (%), Extracurricular Activities, Parental Involvement, Age
X_new_student = np.array([[6, 90, 2, 5, 16]])
X_new_student_norm = scaler.transform(X_new_student)
Y_pred_new_student = sgdr.predict(X_new_student_norm)


fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 10), constrained_layout=True)

axs = axs.flatten()

for index, row in enumerate(np.transpose(data[:, :-1])):
    X_title = row[:1][0]
    X_index = row[1:].astype(float)

    sorted_indexes = np.argsort(X_index)
    X_index_sorted = X_index[sorted_indexes]
    Y_sorted = Y[sorted_indexes]
    Y_pred_sorted = Y_pred[sorted_indexes]

    axs[index].scatter(X_index_sorted, Y_sorted, label=X_title, s=70, color="orangered", edgecolor="black")
    axs[index].scatter(X_index_sorted, Y_pred_sorted, color="blue", s=70, label="Prediction", edgecolor="black")
    axs[index].scatter(X_new_student[0, index], Y_pred_new_student[0], color="yellow", s=70, label="New Student Prediction", edgecolor="black")

    axs[index].set_ylabel(X_title)
    axs[index].set_xlabel("Final Grade (%)")
    axs[index].grid(linestyle="--", color="green")
    axs[index].legend()


axs[-1].axis('off')
plt.show()