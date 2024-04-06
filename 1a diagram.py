import pandas as pd
import matplotlib.pyplot as plt

# Read dataset "Trips_By_Distance.csv" to DataFrame (i.e. df)
df = pd.read_csv("Trips_by_Distance.csv")

# Count the unique values for the "Week" column
df_week_count = df['Week'].nunique()

# Group the average Population Staying at Home per week
average_staying_at_home = df.groupby(by='Week')['Population Staying at Home'].mean()

# Read dataset “Trips_Full_Data.csv” to DataFrame (i.e. df_full)
df_full = pd.read_csv("Trips_Full_Data.csv")

# Count the unique values for the "Week" column in df_full
df_full_week_count = df_full['Week of Date'].nunique()

# Calculate total number of trips for each distance range
df_full['Total Trips'] = df_full[['Trips <1 Mile', 'Trips 1-25 Miles', 'Trips 25-50 Miles', 
                                   'Trips 50-100 Miles', 'Trips 100-250 Miles', 
                                   'Trips 250-500 Miles', 'Trips 500+ Miles']].sum(axis=1)

# Group the average number of people traveling by distance range per week
average_people_traveling_by_distance = df_full.groupby(by='Week of Date')[['Trips <1 Mile', 'Trips 1-25 Miles', 
                                                                   'Trips 25-50 Miles', 'Trips 50-100 Miles', 
                                                                   'Trips 100-250 Miles', 'Trips 250-500 Miles', 
                                                                   'Trips 500+ Miles']].sum() / df_full_week_count

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot the barplot People staying at home vs Week
average_staying_at_home.plot(kind='bar', color='orange', ax=ax1)
ax1.set_xlabel("Week")
ax1.set_ylabel("Average Population Staying at Home")
ax1.set_title("Average Population Staying at Home vs Week")
ax1.grid(True)
ax1.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability

# Plot the barplot People travelling vs Distance
average_people_traveling_by_distance.plot(kind='bar', ax=ax2)
ax2.set_xlabel("Distance Traveled (Miles)")
ax2.set_ylabel("Frequency")
ax2.set_title("People Traveling vs Distance")
ax2.grid(True)
ax2.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()
