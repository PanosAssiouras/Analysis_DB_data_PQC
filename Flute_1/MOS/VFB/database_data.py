import os
import pathlib
import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("display.max_columns", None)
parameter = "VFB_V"
#----------------------------------------------------------------------------------------------#
path = pathlib.Path().absolute()
file_name_path = os.path.join(path, '../../c9140.csv')
metadata_path = os.path.join(path, '../../c8920.csv')
# file_name = os.path.splitext('FET.txt')[0]
#----read and sort data from file exclude nan values ---------------------------------------------------------#
data = pd.read_csv(file_name_path, sep=",", skiprows=0)
data_sorted = data.sort_values(by=['PART_NAME_LABEL'], ascending=True)
data_sorted.reset_index(drop=True, inplace=True)
data_sorted_notnan = data_sorted[(data_sorted[parameter].notna()) & (data_sorted[parameter]!=0)]
#data_sorted_notnan = data_sorted[(data_sorted[parameter].notna())]
data_sorted_notnan.reset_index(drop=True, inplace=True)
data_sorted_notnan.to_csv(r'data_sorted_notnan.csv', index=False)
# ------------------------------------------------------------------------------------------#
# read and sort metadata -------------------------------------------------------------------#
metadata = pd.read_csv(metadata_path, sep=",", skiprows=0)  # loads file with metadata
metadata.dropna(how='all', axis=1, inplace=True)  # drops empty columns
metadata = metadata.sort_values(by=['PART_NAME_LABEL'], ascending=True)
metadata.reset_index(drop=True, inplace=True)
# ------------------------------------------------------------------------------------------------#
locations = []  # set list for location storage
for index in range(len(metadata)):
    # find location from file name
    location = " "
    file_name = metadata['FILE_NAME'].loc[index]
    if type(file_name) == str:
        if file_name.find(".json") != -1:
            location = "HEPHY"
        elif file_name.find(".xml") != -1:
            location = "Demokritos"
        elif file_name.find("Test_HPK") != -1:
            location = "Brown"
        else:
            location = "Perugia"
    locations.append(location)
metadata.insert(len(metadata.columns), "Location", locations, True)  # insert location into metadata file

#------- reduce data in metadata according to what we need ----------------------------------#
meradata_reduced = metadata.iloc[:, 8:]
meradata_reduced = meradata_reduced[meradata_reduced['KIND_OF_HM_STRUCT_ID'] == "MOS_QUARTER"]
#meradata_reduced = meradata_reduced.drop_duplicates(subset=["FILE_NAME"], keep=False)
meradata_reduced = meradata_reduced.drop_duplicates(subset=["FILE_NAME"])
meradata_reduced.reset_index(drop=True, inplace=True)
#print(meradata_reduced)
meradata_reduced.to_csv(r'meradata_reduced.csv', index=False)
#----------------------------------------------------------------------------------------------#
#-------------- merge data with metadata and build file with all information-------------------#
data_merged = pd.merge(data_sorted_notnan, meradata_reduced, how='inner', on=['PART_NAME_LABEL'])
data_merged = data_merged.drop_duplicates(subset=["ID"])
data_merged.dropna(how='all', axis=1, inplace=True)
data_merged = data_merged.sort_values(by=['PART_NAME_LABEL'], ascending=True)
data_merged.reset_index(drop=True, inplace=True)
data_merged.to_csv(r'data_merged.csv', index=False)
# display(data_merged)

# print(data_sorted_notnan)

# print(metadata)
#----------------------------------------------------------------------------------------------#
#---------- take batch number and sensor type from the PART_NAME_LABEL ------------------------#
batch_numbers = []
sensor_types = []
for index in range(len(data_merged)):
    batch_number = data_merged['PART_NAME_LABEL'].loc[index].split("_")[0]
    sensor_type = data_merged['PART_NAME_LABEL'].loc[index].split("_")[2]
    batch_numbers.append(batch_number)
    sensor_types.append(sensor_type)

# Using DataFrame.insert() to add a column
data_merged.insert(0, "batch_number", batch_numbers, True) # batch number inserted to table in separate column
data_merged.insert(1, "Type", sensor_types, True)          # type inserted to table in separate column
data_merged.to_csv(r'data_merged.csv', index=False)
#----------------------------------------------------------------------------------------------#
#------- Filter outliers-----------------------------------------------------------------------#
z_scores = stats.zscore(data_merged[parameter])  # calculate z-scores of `df`
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 50)
new_data_merged = data_merged[filtered_entries]
#----------------------------------------------------------------------------------------------#
#------ reduce data_merged --------------------------------------------------------------------#
data_merged_reduced = new_data_merged[['batch_number', 'Type', parameter, 'Location', 'KIND_OF_HM_SET_ID']]
data_merged_reduced.reset_index(drop=True, inplace=True)
data_merged_reduced.to_csv(parameter+'_all.csv', index=False)

#---------------------- calculate averages and stadard deviation--------------------------------#
print(data_merged_reduced)
data_merged_reduced_mean = data_merged_reduced.groupby(['batch_number', 'Type', 'Location'])[parameter].mean().reset_index(name="Average")
data_merged_reduced_std = data_merged_reduced.groupby(['batch_number', 'Type', 'Location'])[parameter].std().reset_index(name="Std.dev")
data_merged2 = pd.merge(data_merged_reduced_mean, data_merged_reduced_std, how='inner')
data_merged2.rename(columns={'batch_number': 'Name'}, inplace=True)

data_merged2.to_csv(parameter+'_average_values.csv', index=False)
