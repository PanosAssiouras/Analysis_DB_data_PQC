import os
import pathlib
import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("display.max_columns", None)
parameters = ["RC_OHM", "R_OHM"]
search_limits = [[10000, 200000], [0, 500.0]]
measurement = "CBKR_POLY"
flute = "PQC4"
#search_lower_limit = 0.0
#search_upper_limit = 15.0
#----------------------------------------------------------------------------------------------#
path = pathlib.Path().absolute()
database_data_path = os.path.join(path, '../../merged_with_metadata.csv')
#metadata_path = os.path.join(path, '../c8920.csv')
# file_name = os.path.splitext('FET.txt')[0]
#----read and sort data from file exclude nan values ---------------------------------------------------------#
param_number=0
for parameter in parameters:
    data = pd.read_csv(database_data_path, sep=",", skiprows=0)
    # Assuming df is your DataFrame and column_name is the column you want to check for duplicates
    data.drop_duplicates(subset=["FILE_NAME"], inplace=True)
    # Reset the index if needed
    data.reset_index(drop=True, inplace=True)
    data = data[data[parameter].notna()]
    data = data.dropna(axis=1, how="all")
    data = data.loc[data[parameter] != 0]
    # data.reset_index(drop=True, inplace=True)
    data = data.drop(['PART_BARCODE', 'ID', 'KIND_OF_CONDITION_ID', 'CONDITION_DATA_SET_ID',
                      'KIND_OF_PART_ID', 'KIND_OF_CONDITION', 'KIND_OF_PART'], axis=1)
    data = data[(data['KIND_OF_HM_STRUCT_ID'] == measurement) | (data['KIND_OF_HM_STRUCT_ID'] == flute)]
    data = data.loc[(data[parameter] >= search_limits[param_number][0])
                    & (data[parameter] <= search_limits[param_number][1])]
    data.reset_index(drop=True, inplace=True)

    # ----------------------------------------------------------------------------------------------#
    # ---------- take batch number and sensor type from the PART_NAME_LABEL ------------------------#
    batch_numbers = []
    sensor_types = []
    orientations = []
    for index in range(len(data)):
        batch_number = data['PART_NAME_LABEL'].loc[index].split("_")[0]
        sensor_type = data['PART_NAME_LABEL'].loc[index].split("_")[2]
        orientation = data['PART_NAME_LABEL'].loc[index].split("_")[4]
        orientations.append(orientation)
        batch_numbers.append(batch_number)
        sensor_types.append(sensor_type)

    data.insert(0, "batch_number", batch_numbers, True)  # batch number inserted to table in separate column
    data.insert(1, "Type", sensor_types, True)  # type inserted to table in separate column
    data.insert(2, "Orientation", orientations, True)  # orientation West or East inserted to table in separate column
    data = data.replace('2-S', '2S')
    data['RC_OHM']= 1e-3*data['RC_OHM'].abs()
    data.to_csv(r'data.csv', index=False)
    # ----------------------------------------------------------------------------------------------#
    # ------- Filter outliers-----------------------------------------------------------------------#
    z_scores = stats.zscore(data[parameter])  # calculate z-scores of `df`
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 100)
    data_filtered = data[filtered_entries]
    # ----------------------------------------------------------------------------------------------#
    # ------ reduce data_merged --------------------------------------------------------------------#
    data_reduced = data_filtered[
        ['batch_number', 'Type', 'Orientation', 'Location', 'KIND_OF_HM_SET_ID', 'TEMP_SET_DEGC', 'AV_TEMP_DEGC',
         parameter]]
    # data_reduced.rename(columns={'batch_number': 'Name'}, inplace=True)
    data_reduced.reset_index(drop=True, inplace=True)

    data_reduced.to_csv(parameter + '_all.csv', index=False)
    data.to_csv(r'data_reduced.csv', index=False)
    # ---------------------- calculate averages and stadard deviation--------------------------------#

    data_merged_reduced_mean = data_reduced.groupby(['batch_number', 'Type'])[parameter].mean().reset_index(name="Average")
    data_merged_reduced_std = data_reduced.groupby(['batch_number', 'Type'])[parameter].std().reset_index(name="std")
    data_reduced2 = pd.merge(data_merged_reduced_mean, data_merged_reduced_std, how='inner')
    # data_merged2.rename(columns={'batch_number': 'Name'}, inplace=True)

    data_reduced2.to_csv(parameter + '_average_values.csv', index=False)
    param_number+=1