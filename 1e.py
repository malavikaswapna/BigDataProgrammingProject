import pandas as pd
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv("/Users/malavikaswapna/Desktop/Trips_Full_Data.csv")

# Plotting the number of travelers by distance-trips
plt.figure(figsize=(10, 6))

# Plotting trips by distance
plt.plot(df['Week of Date'], df['Trips <1 Mile'], label='<1 Mile')
plt.plot(df['Week of Date'], df['Trips 1-25 Miles'], label='1-25 Miles')
plt.plot(df['Week of Date'], df['Trips 25-50 Miles'], label='25-50 Miles')
plt.plot(df['Week of Date'], df['Trips 50-100 Miles'], label='50-100 Miles')
plt.plot(df['Week of Date'], df['Trips 100-250 Miles'], label='100-250 Miles')
plt.plot(df['Week of Date'], df['Trips 250-500 Miles'], label='250-500 Miles')
plt.plot(df['Week of Date'], df['Trips 500+ Miles'], label='500+ Miles')

# Adding labels and title
plt.xlabel('Week of Date')
plt.ylabel('Number of Travelers')
plt.title('Number of Travelers by Distance-Trips')
plt.legend()

# Display the plot
plt.grid(True)
plt.tight_layout()
plt.show()
