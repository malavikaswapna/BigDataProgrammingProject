import pandas as pd
import matplotlib.pyplot as plt

# Read the dataset
df_trips_by_distance = pd.read_csv("/Users/malavikaswapna/Desktop/Trips_By_Distance.csv")

# Filter rows for 10-25 trips with more than 10,000,000 people
df_10_25_trips = df_trips_by_distance[(df_trips_by_distance['Number of Trips 10-25'] > 10000000)]

# Filter rows for 50-100 trips with more than 10,000,000 people
df_50_100_trips = df_trips_by_distance[(df_trips_by_distance['Number of Trips 50-100'] > 10000000)]

# Extract dates for the filtered rows
dates_10_25_trips = df_10_25_trips['Date']
dates_50_100_trips = df_50_100_trips['Date']

# Plot scatterplot for 10-25 trips
plt.figure(figsize=(10, 6))
plt.scatter(dates_10_25_trips, df_10_25_trips['Number of Trips 10-25'], color='blue', label='10-25 Trips')
plt.xlabel('Date')
plt.ylabel('Number of Trips')
plt.title('Dates vs Number of Trips (10-25 Trips)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()

# Plot scatterplot for 50-100 trips
plt.figure(figsize=(10, 6))
plt.scatter(dates_50_100_trips, df_50_100_trips['Number of Trips 50-100'], color='red', label='50-100 Trips')
plt.xlabel('Date')
plt.ylabel('Number of Trips')
plt.title('Dates vs Number of Trips (50-100 Trips)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()
