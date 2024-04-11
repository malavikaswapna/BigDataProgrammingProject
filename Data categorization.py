import dask.dataframe as dd
import matplotlib.pyplot as plt

# Read dataset "Trips_By_Distance.csv" to DataFrame (df)
df = dd.read_csv("Trips_By_Distance.csv", dtype={'Population Staying at Home': 'float64'})

# Count the unique values for the "Week" column
unique_weeks = df['Week'].nunique().compute()
print("Unique weeks:", unique_weeks)

# Group the average Population Staying at Home per week
avg_staying_at_home = df.groupby(by='Week')['Population Staying at Home'].mean().compute()
print("Average Population Staying at Home per week:")
print(avg_staying_at_home)

# Plot the barplot
plt.figure(figsize=(10, 7))
plt.bar(avg_staying_at_home.index, avg_staying_at_home.values, color='orange')
plt.xlabel("Week")
plt.ylabel("Average Population Staying at Home")
plt.title("Average Population Staying at Home per Week")
plt.grid(True)
plt.show()
