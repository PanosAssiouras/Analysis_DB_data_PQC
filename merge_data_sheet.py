import os
import pathlib
import numpy as np
import pandas as pd
from scipy import stats

path = pathlib.Path().absolute()
IV_data_path = os.path.join(path, './c9120.csv')
CV_data_path = os.path.join(path, './c9140.csv')
TC_data_path = os.path.join(path, './c9160.csv')
metadata_path = os.path.join(path, './c8920.csv')

IV_data = pd.read_csv(IV_data_path, sep=",", skiprows=0)
CV_data = pd.read_csv(CV_data_path, sep=",", skiprows=0)
TC_data = pd.read_csv(TC_data_path, sep=",", skiprows=0)
metadata = pd.read_csv(metadata_path, sep=",", skiprows=0)

print(IV_data)

print(TC_data['V_TH'])
data_merged = pd.concat([IV_data, CV_data, TC_data])
print(data_merged['V_TH'])
data_merged = data_merged.sort_values(by=['PART_ID', 'CONDITION_DATA_SET_ID'], ascending=True).reset_index(drop=True)
data_merged.to_csv('merged.csv', index=False)

metadata = metadata.sort_values(by=['PART_ID', 'CONDITION_DATA_SET_ID'], ascending=True)
print(metadata)
metadata_reduced = metadata.iloc[:, 11:].reset_index(drop=True)
metadata_reduced.to_csv('metadata_reduced.csv', index=False)

#data_merged3 = pd.merge(data_merged2, metadata_reduced)
data_merged2 = pd.concat([data_merged, metadata_reduced],axis=1)
print(data_merged2)
#data_merged2.rename(columns={'batch_number': 'Name'}, inplace=True)
#data_merged2.to_csv(parameter+'_average_values.csv', index=False)
# ------------------------------------------------------------------------------------------------#
locations = []  # set list for location storage
for index in range(len(data_merged2)):
    # find location from file name
    location = " "
    file_name = data_merged2['FILE_NAME'].loc[index]
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
data_merged2.insert(len(data_merged2.columns), "Location", locations, True)  # insert location into metadata file

data_merged2.to_csv('merged_with_metadata.csv', index=False)


