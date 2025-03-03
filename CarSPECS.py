import os
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kagglehub import KaggleDatasetAdapter
from MLRegression import MultipleLinearRegression

# Download latest version
path = kagglehub.dataset_download("rkiattisak/sports-car-prices-dataset")
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
if csv_files:
    df = pd.read_csv(os.path.join(path, csv_files[0]), encoding='latin1')
    print(df.head())
else:
    print("No CSV files found in the dataset directory")

# Print the first few rows of the dataset
print(df.head())

# Convert all values to numeric and check for non-numeric values
print("Before cleaning - unique engine size values:", df["Engine Size (L)"].unique())
df["Engine Size (L)"] = pd.to_numeric(df["Engine Size (L)"], errors='coerce')
df["Horsepower"] = pd.to_numeric(df["Horsepower"], errors='coerce')
df["0-60 MPH Time (seconds)"] = pd.to_numeric(df["0-60 MPH Time (seconds)"], errors='coerce')

# Drop any rows with non-numeric values
df_clean = df.dropna(subset=["Engine Size (L)", "Horsepower", "0-60 MPH Time (seconds)"])

# Use the cleaned data
X = df_clean[["Engine Size (L)", "Horsepower"]].values.astype(float)  # Force float type
y_acceleration_capability = 1 / df_clean["0-60 MPH Time (seconds)"].values.astype(float)  # Force float type

print("X shape:", X.shape)
print("y shape:", y_acceleration_capability.shape)

# Create and fit the model
model = MultipleLinearRegression()
model.fit(X, y_acceleration_capability)
    
# Print coefficients
coeffs = model.get_coefficients()
print(f"Intercept (β0): {coeffs[0][0]:.4f}")
print(f"Coefficient X1 (β1): {coeffs[1][0]:.4f}")
print(f"Coefficient X2 (β2): {coeffs[2][0]:.4f}")
print(f"Equation: {model.get_equation_string()}")
print(f"R-squared: {model.get_r_squared():.4f}")

# Make predictions for the original data points
# predictions = model.predict(X)
# print("Predictions:", predictions.flatten())
    
# Plot the model
fig, ax = model.plot_3d(X, y_acceleration_capability)
plt.title("Car Acceleration Capability")
plt.xlabel("Engine Size (L)")
plt.ylabel("Horsepower")
plt.clabel("Acceleration Capability")
plt.show()

