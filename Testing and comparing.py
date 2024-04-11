import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LinearRegression
from dask.distributed import Client
import multiprocessing
import matplotlib.pyplot as plt

# Add this if statement to fix the multiprocessing error on Windows
if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Read the dataset using Dask DataFrame
    df_full = dd.read_csv("Trips_Full_Data.csv")

    # Data Preparation
    # Handle missing values (if any)
    df_full = df_full.dropna()
    # Feature Selection
    X = df_full[['People Not Staying at Home']].to_dask_array(lengths=True)  # Convert to Dask array
    y = df_full['Trips 1-25 Miles'].to_dask_array(lengths=True)  # Convert to Dask array

    # Split data into train and test sets using Dask
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Parallel Computing Setup
    client = Client()

    # Model Training using parallel processing
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Close the client
    client.close()

    # Track the traveling distance (number of travelers)
    traveling_distance = y_test.compute()

    # Calculate the total number of travelers
    total_travelers = df_full['People Not Staying at Home'].sum().compute()

    # Extrapolate the average distance traveled per person
    average_distance_per_person = y_pred.mean().compute()

    print("Total number of travelers:", total_travelers)
    print("Average distance traveled per person:", average_distance_per_person)

    # Plot the distribution of trip distances for people who are not staying at home
    df_no_home = df_full[df_full['People Not Staying at Home'] > 0]
    plt.figure(figsize=(10, 6))
    plt.hist(df_no_home['Trips 1-25 Miles'].compute(), bins=20, color='blue', alpha=0.7)
    plt.title('Distribution of Trip Distances for People Not Staying at Home')
    plt.xlabel('Trip Distance (1-25 Miles)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()