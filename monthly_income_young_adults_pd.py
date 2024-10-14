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
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.options.display.float_format = '{:,.2f}'.format

training_data = pd.read_csv('training_data.csv')
# training_data = pd.read_csv('training_data_large.csv')

training_data["Monthly Income (MXN)"] = training_data["Monthly Income (MXN)"].astype(float)


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
    "Study Hours": [4, 5, 2, 6, 3, 7],
    "Attendance (%)": [85, 90, 70, 95, 80, 100],
    "Extracurricular Activities": [1, 2, 0, 3, 1, 4],
    "Parental Involvement": [3, 4, 2, 5, 3, 5],
    "Age": [16, 17, 15, 18, 16, 19],
})

X_new_young_adults_data = new_young_adults_data.iloc[:, :].values
X_new_young_adults_data_norm = scaler.transform(X_new_young_adults_data)
Y_new_young_adults_data_predict = sgdr.predict(X_new_young_adults_data_norm)

new_young_adults_data_result = pd.concat([
    new_young_adults_data,
    pd.DataFrame(Y_new_young_adults_data_predict, columns=["Predicted Salary"])
], axis=1)

print(f"Iterations: {sgdr.n_iter_} w: {w}, b: {b}\n")

print("\nTraining Data")
print("=" * 40)
print(training_data)

print("\nPredicted Salaries for Young Adults")
print("=" * 40)
print(new_young_adults_data_result)

print("\nModel Metrics")
print("=" * 40)
mae = mean_absolute_error(Y, Y_pred)
r2 = r2_score(Y, Y_pred)
print(f"mae: {mae:.2f} MXN")
print(f"R-squared (R²): {r2 * 100:,.2f}%")

