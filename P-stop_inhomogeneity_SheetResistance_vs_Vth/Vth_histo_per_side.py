import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D


# Load dataframes from CSV files
df_vth = pd.read_csv('VTH_V_all.csv', delimiter=',')
df_rsh = pd.read_csv('RSH_OHMSQR_all.csv', delimiter=',')

# Merge dataframes on common columns
merged_df = pd.merge(df_vth, df_rsh, on=['batch_number', 'Type', 'Orientation', 'Location', 'KIND_OF_HM_SET_ID', 'TEMP_SET_DEGC', 'AV_TEMP_DEGC'], how='inner')  # Change 'common_column' to the actual common column name
# Exclude entries with Type PSP
merged_df = merged_df[merged_df['Type'] != 'PSP']
# Alternatively, if you want to merge on multiple common columns, you can pass a list of column names:
# merged_df = pd.merge(df_vth, df_rsh, on=['common_column1', 'common_column2'], how='inner')

# Print the merged dataframe
print(merged_df)

# Create separate dataframes based on conditions
ww_right_df = merged_df[(merged_df['Orientation'] == 'WW') & (merged_df['KIND_OF_HM_SET_ID'] == 'Right')]
ww_left_df = merged_df[(merged_df['Orientation'] == 'WW') & (merged_df['KIND_OF_HM_SET_ID'] == 'Left')]
ee_right_df = merged_df[(merged_df['Orientation'] == 'EE') & (merged_df['KIND_OF_HM_SET_ID'] == 'Right')]
ee_left_df = merged_df[(merged_df['Orientation'] == 'EE') & (merged_df['KIND_OF_HM_SET_ID'] == 'Left')]

# Create histograms
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.hist(ww_right_df['VTH_V'], bins=20, color='blue', alpha=0.7)
plt.title('WW and Right')
plt.xlabel('VTH_V')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
plt.hist(ww_left_df['VTH_V'], bins=20, color='green', alpha=0.7)
plt.title('WW and Left')
plt.xlabel('VTH_V')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
plt.hist(ee_right_df['VTH_V'], bins=20, color='red', alpha=0.7)
plt.title('EE and Right')
plt.xlabel('VTH_V')
plt.ylabel('Frequency')

plt.subplot(2, 2, 4)
plt.hist(ee_left_df['VTH_V'], bins=20, color='orange', alpha=0.7)
plt.title('EE and Left')
plt.xlabel('VTH_V')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

