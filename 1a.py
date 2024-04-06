import pandas as pd

# Load the datasets
trips_by_distance = pd.read_csv("Trips_by_Distance.csv")
trips_full_data = pd.read_csv("Trips_Full_Data.csv")

# Calculate the average number of people staying at home per week
avg_people_staying_home_per_week = trips_by_distance.groupby('Week')['Population Staying at Home'].mean()

# Calculate the average number of trips for each distance category when people don't stay at home
avg_trips_not_staying_home_by_distance = trips_full_data.groupby('Week of Date')[['Trips <1 Mile', 'Trips 1-25 Miles', 'Trips 25-50 Miles', 'Trips 50-100 Miles', 'Trips 100-250 Miles', 'Trips 250-500 Miles', 'Trips 500+ Miles']].mean()

# Print the results
print("Average number of people staying at home per week:")
print(avg_people_staying_home_per_week)

print("\nAverage number of trips for each distance category when people don't stay at home:")
print(avg_trips_not_staying_home_by_distance)
