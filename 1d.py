import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Read datasets
df_full = pd.read_csv("Trips_Full_Data.csv")
df = pd.read_csv("Trips_by_Distance.csv")

# Extract week number from 'Week of Date' column in df_full
df_full['Week of Date'] = df_full['Week of Date'].str.extract('(\d+)')
df_full['Week of Date'] = pd.to_numeric(df_full['Week of Date'])

# Merge the DataFrames on the 'Week of Date' column
merged_df = pd.merge(df_full, df, left_on='Week of Date', right_on='Week', how='inner')

# Print the merged DataFrame
print(merged_df.head())

# Drop rows with missing values
merged_df.dropna(subset=['Trips 1-25 Miles', 'Number of Trips 5-10', 'Number of Trips 10-25'], inplace=True)

# Select appropriate features and target
x = merged_df[['Trips 1-25 Miles', 'Trips 25-100 Miles']].values  # Features
y = merged_df[['Number of Trips 5-10', 'Number of Trips 10-25']].values  # Target

# Flatten target for Linear Regression (assuming we're predicting one target at a time)
y_5_10 = y[:, 0]  # Target for Number of Trips 5-10
y_10_25 = y[:, 1]  # Target for Number of Trips 10-25

# Linear Regression for Number of Trips 5-10
model_linear_5_10 = LinearRegression()
model_linear_5_10.fit(x, y_5_10)
linear_r_squared_5_10 = model_linear_5_10.score(x, y_5_10)

# Linear Regression for Number of Trips 10-25
model_linear_10_25 = LinearRegression()
model_linear_10_25.fit(x, y_10_25)
linear_r_squared_10_25 = model_linear_10_25.score(x, y_10_25)

# Polynomial Regression for Number of Trips 5-10
poly_features = PolynomialFeatures(degree=2)
x_poly = poly_features.fit_transform(x)
model_poly_5_10 = LinearRegression()
model_poly_5_10.fit(x_poly, y_5_10)
poly_r_squared_5_10 = model_poly_5_10.score(x_poly, y_5_10)

# Polynomial Regression for Number of Trips 10-25
model_poly_10_25 = LinearRegression()
model_poly_10_25.fit(x_poly, y_10_25)
poly_r_squared_10_25 = model_poly_10_25.score(x_poly, y_10_25)

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train models with training data
model_linear_5_10.fit(x_train, y_train[:, 0])  # Train Linear Regression for Number of Trips 5-10
model_linear_10_25.fit(x_train, y_train[:, 1])  # Train Linear Regression for Number of Trips 10-25
model_poly_5_10.fit(poly_features.fit_transform(x_train), y_train[:, 0])  # Train Polynomial Regression for Number of Trips 5-10
model_poly_10_25.fit(poly_features.fit_transform(x_train), y_train[:, 1])  # Train Polynomial Regression for Number of Trips 10-25

# Evaluate models
linear_train_r_squared_5_10 = model_linear_5_10.score(x_train, y_train[:, 0])
linear_train_r_squared_10_25 = model_linear_10_25.score(x_train, y_train[:, 1])
poly_train_r_squared_5_10 = model_poly_5_10.score(poly_features.fit_transform(x_train), y_train[:, 0])
poly_train_r_squared_10_25 = model_poly_10_25.score(poly_features.fit_transform(x_train), y_train[:, 1])

# Predictions
y_pred_linear_5_10 = model_linear_5_10.predict(x_test)
y_pred_linear_10_25 = model_linear_10_25.predict(x_test)
y_pred_poly_5_10 = model_poly_5_10.predict(poly_features.fit_transform(x_test))
y_pred_poly_10_25 = model_poly_10_25.predict(poly_features.fit_transform(x_test))

# Evaluate predictions
linear_test_r_squared_5_10 = r2_score(y_test[:, 0], y_pred_linear_5_10)
linear_test_r_squared_10_25 = r2_score(y_test[:, 1], y_pred_linear_10_25)
poly_test_r_squared_5_10 = r2_score(y_test[:, 0], y_pred_poly_5_10)
poly_test_r_squared_10_25 = r2_score(y_test[:, 1], y_pred_poly_10_25)

print("Linear Regression Train R-squared (Number of Trips 5-10):", linear_train_r_squared_5_10)
print("Linear Regression Train R-squared (Number of Trips 10-25):", linear_train_r_squared_10_25)
print("Polynomial Regression Train R-squared (Number of Trips 5-10):", poly_train_r_squared_5_10)
print("Polynomial Regression Train R-squared (Number of Trips 10-25):", poly_train_r_squared_10_25)
print("Linear Regression Test R-squared (Number of Trips 5-10):", linear_test_r_squared_5_10)
print("Linear Regression Test R-squared (Number of Trips 10-25):", linear_test_r_squared_10_25)
print("Polynomial Regression Test R-squared (Number of Trips 5-10):", poly_test_r_squared_5_10)
print("Polynomial Regression Test R-squared (Number of Trips 10-25):", poly_test_r_squared_10_25)