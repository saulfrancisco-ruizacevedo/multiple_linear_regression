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
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
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


X = data[1:, :-1]
Y = data[1:, -1].astype(float)

# Normalize data
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

sgdr = SGDRegressor(max_iter=10_000)
sgdr.fit(X_norm, Y)
Y_pred = sgdr.predict(X_norm)

w = sgdr.coef_
b = sgdr.intercept_
print(f"Regression Data w:{w}, b:{b}")


# Study Hours, Attendance (%), Extracurricular Activities, Parental Involvement, Age
X_new_student = np.array([[6, 90, 2, 5, 16]])
X_new_student_norm = scaler.transform(X_new_student)
Y_pred_new_student = sgdr.predict(X_new_student_norm)

# Displaying student data in a table format
headers = data[:1, :-1][0]
student_data = [X_new_student[0].tolist()]
print("\nStudent Data:")
print(tabulate(student_data, headers, tablefmt="pretty"))
print(f"Expected final grade: {Y_pred_new_student[0]:.2f}/100")

print("\nModel Metrics")
mae = mean_absolute_error(Y, Y_pred)
r2 = r2_score(Y, Y_pred)
print(f"mae: {mae:.2f} points")
print(f"R-squared (R²): {r2 * 100:,.2f}%")