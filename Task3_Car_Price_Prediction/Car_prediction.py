# CodeAlpha Data Science Internship
# Task 3 - Car Price Prediction with Machine Learning

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("car data.csv")

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())


print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())



df.columns = df.columns.str.strip()


df = df.drop_duplicates()

print("\nDataset shape after removing duplicates:")
print(df.shape)



current_year = 2026

df["Car_Age"] = current_year - df["Year"]

print("\nCar age feature created.")



X = df[
    [
        "Car_Name",
        "Year",
        "Present_Price",
        "Driven_kms",
        "Fuel_Type",
        "Selling_type",
        "Transmission",
        "Owner",
        "Car_Age"
    ]
]

y = df["Selling_Price"]


categorical_columns = [
    "Car_Name",
    "Fuel_Type",
    "Selling_type",
    "Transmission"
]

numerical_columns = [
    "Year",
    "Present_Price",
    "Driven_kms",
    "Owner",
    "Car_Age"
]



preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)


model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)


pipeline.fit(X_train, y_train)

print("\nModel training completed successfully!")



y_pred = pipeline.predict(X_test)

print("\nFirst 10 predicted prices:")
print(y_pred[:10])


mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)


print("\n========== MODEL EVALUATION ==========")

print(f"Mean Absolute Error: {mae:.2f}")

print(f"Mean Squared Error: {mse:.2f}")

print(f"Root Mean Squared Error: {rmse:.2f}")

print(f"R2 Score: {r2:.2f}")


plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Selling Price")

plt.ylabel("Predicted Selling Price")

plt.title("Actual vs Predicted Car Prices")

plt.tight_layout()

plt.show()



plt.figure(figsize=(8, 6))

plt.scatter(
    df["Present_Price"],
    df["Selling_Price"]
)

plt.xlabel("Present Price")

plt.ylabel("Selling Price")

plt.title("Present Price vs Selling Price")

plt.tight_layout()

plt.show()


plt.figure(figsize=(8, 6))

plt.scatter(
    df["Car_Age"],
    df["Selling_Price"]
)

plt.xlabel("Car Age")

plt.ylabel("Selling Price")

plt.title("Car Age vs Selling Price")

plt.tight_layout()

plt.show()


plt.figure(figsize=(8, 5))

sns.boxplot(
    x="Fuel_Type",
    y="Selling_Price",
    data=df
)

plt.xlabel("Fuel Type")

plt.ylabel("Selling Price")

plt.title("Selling Price by Fuel Type")

plt.tight_layout()

plt.show()


plt.figure(figsize=(8, 5))

sns.boxplot(
    x="Transmission",
    y="Selling_Price",
    data=df
)

plt.xlabel("Transmission")

plt.ylabel("Selling Price")

plt.title("Selling Price by Transmission")

plt.tight_layout()

plt.show()


new_car = pd.DataFrame([
    {
        "Car_Name": "ritz",
        "Year": 2018,
        "Present_Price": 6.0,
        "Driven_kms": 20000,
        "Fuel_Type": "Petrol",
        "Selling_type": "Dealer",
        "Transmission": "Manual",
        "Owner": 0,
        "Car_Age": 2026 - 2018
    }
])


predicted_price = pipeline.predict(new_car)


print("\n========== NEW CAR PREDICTION ==========")

print(
    f"Predicted Selling Price: "
    f"{predicted_price[0]:.2f} lakhs"
)

print("\n======================================")
print("CAR PRICE PREDICTION COMPLETED")
print("======================================")

print("Algorithm: Random Forest Regression")

print(f"R2 Score: {r2:.2f}")

print(
    f"Predicted Price: "
    f"{predicted_price[0]:.2f} lakhs"
)