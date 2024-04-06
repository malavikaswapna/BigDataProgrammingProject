import pandas as pd
import dask.dataframe as dd
import time
import matplotlib.pyplot as plt

# Define number of processors
n_processors = [10, 20]
n_processors_time = {}  # Define n_processors_time dictionary

# Read the dataset
df = pd.read_csv("Trips_By_Distance.csv")

for processor in n_processors:
    start_time = time.time()
    
    # Perform the computations using Pandas
    # Calculate the average number of people staying at home per week
    avg_people_staying_home_per_week = df.groupby('Week')['Population Staying at Home'].mean()
    
    # Filter rows for 10-25 trips with more than 10,000,000 people
    df_10_25_trips = df[df['Number of Trips 10-25'] > 10000000]
    dates_10_25_trips = df_10_25_trips['Date']
    
    # Filter rows for 50-100 trips with more than 10,000,000 people
    df_50_100_trips = df[df['Number of Trips 50-100'] > 10000000]
    dates_50_100_trips = df_50_100_trips['Date']
    
    # Calculate the time taken for computation using Pandas
    pandas_time = time.time() - start_time
    
    # Perform the computations using Dask
    start_time = time.time()
    # Convert pandas DataFrame to Dask DataFrame
    ddf = dd.from_pandas(df, npartitions=processor)
    
    # Calculate the average number of people staying at home per week using Dask
    avg_people_staying_home_per_week_dask = ddf.groupby('Week')['Population Staying at Home'].mean().compute()
    
    # Filter rows for 10-25 trips with more than 10,000,000 people using Dask
    ddf_10_25_trips = ddf[ddf['Number of Trips 10-25'] > 10000000]
    dates_10_25_trips_dask = ddf_10_25_trips['Date'].compute()
    
    # Filter rows for 50-100 trips with more than 10,000,000 people using Dask
    ddf_50_100_trips = ddf[ddf['Number of Trips 50-100'] > 10000000]
    dates_50_100_trips_dask = ddf_50_100_trips['Date'].compute()
    
    # Calculate the time taken for computation using Dask
    dask_time = time.time() - start_time
    
    # Store the computation time in the dictionary
    n_processors_time[processor] = {'Pandas': pandas_time, 'Dask': dask_time}


# Print the time taken for computations
for processor in n_processors_time:
    print(f"Number of Processors: {processor}")
    print(f"Time taken for Pandas (in seconds): {n_processors_time[processor]['Pandas']}")
    print(f"Time taken for Dask (in seconds): {n_processors_time[processor]['Dask']}")
    print()
