# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Prepare the data.
2. Initialize the model
3. Train the model
4. The result is predicted and evaluated

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Elfreeda Jesusha J
RegisterNumber:  212224040084
*/
```
```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score
housing=fetch_california_housing(as_frame=True)
df=housing.frame
X=df.drop(columns=["MedHouseVal","AveOccup"])
y=df[["MedHouseVal","AveOccup"]]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
price_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
occupants_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
price_model.fit(X_train_scaled, y_train["MedHouseVal"])
occupants_model.fit(X_train_scaled, y_train["AveOccup"])
y_pred_price = price_model.predict(X_test_scaled)
y_pred_occupants=occupants_model.predict(X_test_scaled)
print("House Price Prediction Performance:")
print("MSE:", mean_squared_error(y_test["MedHouseVal"], y_pred_price))
print("R2 Score:", r2_score(y_test["MedHouseVal"], y_pred_price))
print("\nOccupants Prediction Performance:")
print("MSE:", mean_squared_error(y_test["AveOccup"], y_pred_occupants))
print("R2 Score:", r2_score(y_test["AveOccup"], y_pred_occupants))
sample = X_test.iloc[0:1]  
sample_scaled = scaler.transform(sample)
pred_price = price_model.predict(sample_scaled)[0]
pred_occupants=occupants_model.predict(sample_scaled)[0]
print("\nSample Input Features:\n", sample)
print(f"\nPredicted House Price: {pred_price:.2f}")
print(f"Predicted Number of Occupants: {pred_occupants:.2f}")

```

## Output:
<img width="413" height="117" alt="exp4ml" src="https://github.com/user-attachments/assets/3bd0f9b9-fba5-4077-b9fb-83f6868c7d48" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
