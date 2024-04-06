import pandas as pd
import dask.dataframe as dd
import time
import matplotlib.pyplot as plt

# Define file path
file_path = "/Users/malavikaswapna/Desktop/Trips_by_Distance.csv"

# Define number of processors to test
n_processors = [10, 20]

# Dictionary to store computation times for each number of processors
n_processors_time = {}

# Read the CSV file using Dask
ddf_dask = dd.read_csv(file_path, assume_missing=True)

# Process data and measure computation time for each number of processors
for processor in n_processors:
    start_time = time.time()
    
    # Process data for question (a) using Dask
    average_people_staying_at_home_per_week_dask = ddf_dask.groupby('Week')['Population Staying at Home'].mean().compute(num_workers=processor)
    
    # Process data for question (b) using Dask
    ddf_10_25_trips_dask = ddf_dask[ddf_dask['Number of Trips 10-25'] > 10000000]['Date']
    ddf_50_100_trips_dask = ddf_dask[ddf_dask['Number of Trips 50-100'] > 10000000]['Date']
    
    # Measure computation time
    computation_time = time.time() - start_time
    
    # Store computation time in the dictionary
    n_processors_time[processor] = computation_time

# Plot computation times
plt.bar(n_processors_time.keys(), n_processors_time.values(), color='skyblue')
plt.xlabel('Number of Processors')
plt.ylabel('Computation Time (seconds)')
plt.title('Computation Time vs Number of Processors')
plt.show()

# Print computation times for each number of processors
for processor, time_taken in n_processors_time.items():
    print(f"Number of Processors: {processor}, Computation Time: {time_taken} seconds")
