import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import argparse
import json

# Load and Preprocess the Bareilly Dataset for 2019, 2020, and 2021

# Step 1: Parse command-line arguments
parser = argparse.ArgumentParser(description='Preprocess Bareilly Dataset')
parser.add_argument('--files', nargs='+', required=True, help='List of file paths for the datasets (e.g. 2019, 2020, 2021)')
parser.add_argument('--output_path', required=True)
args = parser.parse_args()

# Load the Dataset
files = args.files
all_data = []


# Load each CSV file and append to the list
for file in files:
    df = pd.read_csv(file)
    all_data.append(df)

# Concatenate all years into a single DataFrame
data = pd.concat(all_data, axis=0)



# Step 2: Parse the Date Column
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data.dropna(subset=['date'], inplace=True)  # Remove rows with invalid dates
data.set_index('date', inplace=True)

#################################################################################
# Take only those rows where column 'meter' has values 'B01', 'BR03', or 'BR04'
# data = data[data['meter'].isin(['BR02', 'BR03', 'BR05'])]
##################################################################################


# Step 3: Drop Unnecessary Columns
# Drop columns that are not relevant for the forecasting task
columns_to_drop = ['severerisk', 'icon', 'stations', 'description', 'name']  # Update as necessary
data.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')


# Rename the 'meter' column to 'unique_id'
data.rename(columns={'meter': 'unique_id'}, inplace=True)

# Set the target column to be the new 'unique_id'
target_column = 'unique_id'

# Reorder columns to have 'unique_id' as the second column
columns = [col for col in data.columns if col != target_column]  # All columns except 'unique_id'
columns.insert(1, target_column)  # Insert 'unique_id' at the second position
data = data[columns]


# Step 4: Handle Missing Values
# Fill missing values with 0 or use interpolation
data.fillna(0, inplace=True)

#sunrise handling
# Convert the 'sunrise' column to datetime format
data['sunrise'] = pd.to_datetime(data['sunrise'], errors='coerce')
# Drop rows where 'sunrise' could not be parsed properly
data.dropna(subset=['sunrise'], inplace=True)
# Convert sunrise time to seconds since midnight
data['sunrise_seconds'] = data['sunrise'].dt.hour * 3600 + data['sunrise'].dt.minute * 60 + data['sunrise'].dt.second
# Normalize to a range of [0, 1] (assuming 86400 seconds in a day)
data['sunrise_normalized'] = data['sunrise_seconds'] / 86400
# Apply sine and cosine transformations
data['sunrise_sin'] = np.sin(2 * np.pi * data['sunrise_normalized'])
data['sunrise_cos'] = np.cos(2 * np.pi * data['sunrise_normalized'])
# Drop original columns if no longer needed
data.drop(columns=['sunrise', 'sunrise_seconds', 'sunrise_normalized'], inplace=True)


#sunset handling
# Convert the 'sunset' column to datetime format
data['sunset'] = pd.to_datetime(data['sunset'], errors='coerce')
# Drop rows where 'sunset' could not be parsed properly
data.dropna(subset=['sunset'], inplace=True)
# Convert sunset time to seconds since midnight
data['sunset_seconds'] = data['sunset'].dt.hour * 3600 + data['sunset'].dt.minute * 60 + data['sunset'].dt.second
# Normalize to a range of [0, 1] (assuming 86400 seconds in a day)
data['sunset_normalized'] = data['sunset_seconds'] / 86400
# Apply sine and cosine transformations
data['sunset_sin'] = np.sin(2 * np.pi * data['sunset_normalized'])
data['sunset_cos'] = np.cos(2 * np.pi * data['sunset_normalized'])
# Drop original columns if no longer needed
data.drop(columns=['sunset', 'sunset_seconds', 'sunset_normalized'], inplace=True)


# Step 7: Process Conditions Column
# Split conditions into individual components
data['conditions_split'] = data['conditions'].str.split(', ')

# Create new columns for each condition
all_conditions = set(cond for sublist in data['conditions_split'].dropna() for cond in sublist)

conditions_mapping = {}
for i, condition in enumerate(all_conditions):
    data[condition] = data['conditions_split'].apply(lambda x: 1 if condition in x else 0)
    conditions_mapping[condition] = i

# Drop the original 'conditions' and 'conditions_split' columns
data.drop(columns=['conditions', 'conditions_split'], inplace=True)

# Save conditions encoding to JSON file
conditions_mapping_path = os.path.join(args.output_path, 'conditions_mapping.json')
with open(conditions_mapping_path, 'w') as f:
    json.dump(conditions_mapping, f)

# Step 5: Process Preciptype Column
# Handle 'preciptype' column which has either 'rain' or is empty
data['preciptype_rain'] = data['preciptype'].apply(lambda x: 1 if x == 'rain' else 0)
# Drop the original 'preciptype' column
data.drop(columns=['preciptype'], inplace=True)

# # Step 6.1: Encode 'meter' Column and Save Mapping
# meter_mapping = {meter: i for i, meter in enumerate(data['meter'].unique())}
# data['meter_encoded'] = data['meter'].map(meter_mapping)

# # Insert 'meter_encoded' column at the same position where 'meter' was
# data.insert(data.columns.get_loc('meter'), 'meter_encoded', data.pop('meter_encoded'))

# # Save meter encoding to JSON file
# meter_mapping_path = os.path.join(args.output_path, 'meter_mapping.json')
# with open(meter_mapping_path, 'w') as f:
#     json.dump(meter_mapping, f)

# data.drop(columns=['meter'], inplace=True)


# Step 6: Move Target Column to the Last Position
# Move 't_kWh' column to the last position
target_column = 't_kWh'
columns = [col for col in data.columns if col != target_column] + [target_column]
data = data[columns]

# # Step 6.1: Encode Column Names as Numbers and Save Mapping
# # Exclude 'date', 'meter', and the last column for mapping
# columns_to_map = [col for col in data.columns if col not in ['date', 'unique_id']][:-1]
# column_mapping = {col: i for i, col in enumerate(columns_to_map, start=0)}
# data.rename(columns=column_mapping, inplace=True)

# # Save column name to number correspondence
# mapping_output_path = os.path.join(args.output_path, 'column_mapping.json')
# os.makedirs(os.path.dirname(mapping_output_path), exist_ok=True)
# with open(mapping_output_path, 'w') as f:
#     json.dump(column_mapping, f)



# Save the updated DataFrame to a new CSV file
output_csv_path = os.path.join(args.output_path, 'city_data.csv')
data.to_csv(output_csv_path, index=True)
