
import pandas as pd

# Read the dataset
df = pd.read_csv("/Users/malavikaswapna/Desktop/Trips_by_Distance.csv")

# Filter rows for 10-25 trips with more than 10,000,000 people
df_10_25_trips = df[df['Number of Trips 10-25'] > 10000000]
dates_10_25_trips = df_10_25_trips['Date']

# Filter rows for 50-100 trips with more than 10,000,000 people
df_50_100_trips = df[df['Number of Trips 50-100'] > 10000000]
dates_50_100_trips = df_50_100_trips['Date']

print("Dates where more than 10,000,000 people conducted 10-25 trips:")
print(dates_10_25_trips)

print("\nDates where more than 10,000,000 people conducted 50-100 trips:")
print(dates_50_100_trips)
