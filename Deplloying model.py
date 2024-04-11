import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder="/Users/malavikaswapna/Desktop/MyProject/templates")

# Read datasets
df_full = pd.read_csv("Trips_Full_Data.csv")
df = pd.read_csv("Trips_by_Distance.csv")

# Extract week number from 'Week of Date' column in df_full
df_full['Week of Date'] = df_full['Week of Date'].str.extract('(\d+)')
df_full['Week of Date'] = pd.to_numeric(df_full['Week of Date'])

# Merge the DataFrames on the 'Week of Date' column
merged_df = pd.merge(df_full, df, left_on='Week of Date', right_on='Week', how='inner')

# Drop rows with missing values
merged_df.dropna(subset=['Trips 1-25 Miles', 'Number of Trips 5-10', 'Number of Trips 10-25'], inplace=True)

# Select appropriate features and target
x = merged_df[['Trips 1-25 Miles', 'Trips 25-100 Miles']].values  # Features
y = merged_df[['Number of Trips 5-10', 'Number of Trips 10-25']].values  # Target

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train Linear Regression models
model_linear_5_10 = LinearRegression()
model_linear_5_10.fit(x_train, y_train[:, 0])

model_linear_10_25 = LinearRegression()
model_linear_10_25.fit(x_train, y_train[:, 1])

# Train Polynomial Regression models
poly_features = PolynomialFeatures(degree=2)
x_train_poly = poly_features.fit_transform(x_train)

model_poly_5_10 = LinearRegression()
model_poly_5_10.fit(x_train_poly, y_train[:, 0])

model_poly_10_25 = LinearRegression()
model_poly_10_25.fit(x_train_poly, y_train[:, 1])

# Evaluate models on the testing set
linear_r_squared_5_10 = model_linear_5_10.score(x_test, y_test[:, 0])
linear_r_squared_10_25 = model_linear_10_25.score(x_test, y_test[:, 1])

x_test_poly = poly_features.transform(x_test)
poly_r_squared_5_10 = model_poly_5_10.score(x_test_poly, y_test[:, 0])
poly_r_squared_10_25 = model_poly_10_25.score(x_test_poly, y_test[:, 1])

# Save the trained models to disk
joblib.dump(model_linear_5_10, "model_linear_5_10.pkl")
joblib.dump(model_linear_10_25, "model_linear_10_25.pkl")
joblib.dump(model_poly_5_10, "model_poly_5_10.pkl")
joblib.dump(model_poly_10_25, "model_poly_10_25.pkl")

# Load the trained models
model_linear_5_10 = joblib.load("model_linear_5_10.pkl")
model_linear_10_25 = joblib.load("model_linear_10_25.pkl")
model_poly_5_10 = joblib.load("model_poly_5_10.pkl")
model_poly_10_25 = joblib.load("model_poly_10_25.pkl")

@app.route("/")
def home():
    return render_template("index.html", linear_r_squared_5_10=linear_r_squared_5_10,
                           linear_r_squared_10_25=linear_r_squared_10_25,
                           poly_r_squared_5_10=poly_r_squared_5_10,
                           poly_r_squared_10_25=poly_r_squared_10_25)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    trips_1_25_miles = float(data["trips-1-25-miles"])
    trips_25_100_miles = float(data["trips-25-100-miles"])

    # Make predictions using the loaded models
    prediction_linear_5_10 = model_linear_5_10.predict([[trips_1_25_miles, trips_25_100_miles]])[0]
    prediction_linear_10_25 = model_linear_10_25.predict([[trips_1_25_miles, trips_25_100_miles]])[0]

    poly_features = PolynomialFeatures(degree=2)
    features_poly = poly_features.fit_transform([[trips_1_25_miles, trips_25_100_miles]])
    prediction_poly_5_10 = model_poly_5_10.predict(features_poly)[0]
    prediction_poly_10_25 = model_poly_10_25.predict(features_poly)[0]

    return jsonify({
        "Number of Trips 5-10 (Linear Regression) Prediction": prediction_linear_5_10,
        "Number of Trips 10-25 (Linear Regression) Prediction": prediction_linear_10_25,
        "Number of Trips 5-10 (Polynomial Regression) Prediction": prediction_poly_5_10,
        "Number of Trips 10-25 (Polynomial Regression) Prediction": prediction_poly_10_25
    })

if __name__ == "__main__":
    app.run()
