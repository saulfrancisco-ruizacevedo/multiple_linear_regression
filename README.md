# House Price Prediction using multiple linear regression

 This project uses a linear regression model to predict house prices based on features such as area, number of bedrooms, and age. The Stochastic Gradient Descent (SGD) algorithm is used to train the model, and results are visualized with graphs.

 ## Contents
- **Library Imports**
- **Data Definition**
- **Data Normalization**
- **Data Normalization** 
- **Model Training** 
- **Price Prediction** 
- **Results Visualization**  
## Requirements  
Make sure you have Python 3.x installed along with the following libraries:  
- NumPy 
- Matplotlib 
- scikit-learn  

You can install them using pip:
```bash 
pip install numpy matplotlib scikit-learn
```


## Usage

1.  Clone this repository or download the file containing the code.
2.  Run the script in your Python environment:

```bash 
python3 multiple_linear_regression_sklearn.py
```

3.  The script will print the predicted price for a new house and display graphs of house prices based on their features.

## Example

The code includes a dataset of houses structured as follows:

-   **Area (m²)**
-   **Bedrooms**
-   **Age (years)**
-   **Price (thousands of pesos)**

For example, the initial dataset includes houses with characteristics such as:


| Area | Bedrooms | Age | Price | 
|------|----------|-----|-------| 
| 50   | 1        | 5   | 200   | 
| 60   | 2        | 10  | 250   | 
| 70   | 2        | 3   | 280   | 
| 80   | 3        | 8   | 300   | 
| 90   | 3        | 15  | 320   |

The script predicts the price for a new house with the following characteristics:

-   Area: 125 m²
-   Bedrooms: 4
-   Age: 3 years

The script output will include something like:

```bash 
Predicted price for new house with an Area of 125 m², 4 bedrooms, and 3 years is: 225,000.00 mxn
```



## Visualization

Graphs are generated showing:

-   House prices based on area
-   House prices based on the number of bedrooms
-   House prices based on age

The price of the new house is highlighted in the graphs.
