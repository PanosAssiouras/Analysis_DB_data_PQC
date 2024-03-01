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

# Create a scatter plot with hue encoding
sns.scatterplot(data=merged_df, x='VTH_V', y='RSH_OHMSQR', hue='Orientation', style='KIND_OF_HM_SET_ID', palette='viridis')
plt.xlabel('VTH_V')
plt.ylabel('RSH_OHMSQR')
plt.title('Scatter Plot of VTH_V vs RSH_OHMSQR')
plt.legend(title='Orientation')
plt.show()



# Create separate dataframes based on conditions
ww_right_df = merged_df[(merged_df['Orientation'] == 'WW') & (merged_df['KIND_OF_HM_SET_ID'] == 'Right')]
ww_left_df = merged_df[(merged_df['Orientation'] == 'WW') & (merged_df['KIND_OF_HM_SET_ID'] == 'Left')]
ee_right_df = merged_df[(merged_df['Orientation'] == 'EE') & (merged_df['KIND_OF_HM_SET_ID'] == 'Right')]
ee_left_df = merged_df[(merged_df['Orientation'] == 'EE') & (merged_df['KIND_OF_HM_SET_ID'] == 'Left')]

# Create histograms
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.hist(ww_right_df['VTH_V'], bins=20, color='red', alpha=0.7)
plt.title('WW and Right')
plt.xlabel('VTH_V')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
plt.hist(ww_left_df['VTH_V'], bins=20, color='blue', alpha=0.7)
plt.title('WW and Left')
plt.xlabel('VTH_V')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
plt.hist(ee_right_df['VTH_V'], bins=20, color='green', alpha=0.7)
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



# Calculate standard deviation for each batch
std_dev_per_batch = merged_df.groupby('batch_number').agg({'VTH_V': 'std', 'RSH_OHMSQR': 'std'})

print(std_dev_per_batch)
print(max(std_dev_per_batch['VTH_V']), min(std_dev_per_batch['VTH_V']), std_dev_per_batch['VTH_V'].mean())


threshold = std_dev_per_batch.median()

# Identify batches with very high deviation
batches_with_high_deviation = std_dev_per_batch[(std_dev_per_batch['VTH_V'] > threshold['VTH_V']) | (std_dev_per_batch['RSH_OHMSQR'] > threshold['RSH_OHMSQR'])]

print("Batches with very high deviation:")
print(batches_with_high_deviation)

# Threshold for high standard deviation
# Calculate standard deviation for all columns

# Calculate threshold as two times the average value of the standard deviation
threshold1 = 2
threshold2 = 2
# Mark points with high standard deviation
merged_df['High_STD'] = ((merged_df['batch_number'].map(std_dev_per_batch['VTH_V']) > threshold['VTH_V']) &
                         (merged_df['batch_number'].map(std_dev_per_batch['RSH_OHMSQR']) > threshold['RSH_OHMSQR']))

merged_df['Low_STD'] = ((merged_df['batch_number'].map(std_dev_per_batch['VTH_V']) < threshold['VTH_V']) &
                         (merged_df['batch_number'].map(std_dev_per_batch['RSH_OHMSQR'])< threshold['RSH_OHMSQR']))


# Define custom labels for legend
custom_labels = {'WW_right': 'WW_right', 'WW_left': 'WW_left', 'EE_right': 'EE_right', 'EE_left': 'EE_left'}

# Create two plots for High_STD and Low_STD
plt.figure(figsize=(12, 6))



# Create two plots for High_STD and Low_STD
plt.figure(figsize=(12, 6))


custom_palette = ["#FF5733", "#5733FF", "#FFFF33"]  # Example custom colors
sns.set_palette(custom_palette)
# Plot for High_STD

# Custom color palette
custom_palette2 = {"WW_left": "blue", "WW_right": "green", "EE_left": "red", "EE_right": "orange"}

plt.subplot(1, 2, 1)

custom_hue_order = {
    "WW": {"right": 0, "left": 1},
    "EE": {"right": 2, "left": 3}
}

data_WW_Right = merged_df[(merged_df['KIND_OF_HM_SET_ID']=='Right') & (merged_df['Orientation']=='WW')]
high_std_plot = sns.scatterplot(data=data_WW_Right[data_WW_Right['High_STD']], x='VTH_V', y='RSH_OHMSQR', color='red', label="WW_Right")
data_WW_Left = merged_df[(merged_df['KIND_OF_HM_SET_ID']=='Left') & (merged_df['Orientation']=='WW')]
high_std_plot = sns.scatterplot(data=data_WW_Left[data_WW_Left['High_STD']], x='VTH_V', y='RSH_OHMSQR', color='blue', label="WW_Left")
data_EE_Left = merged_df[(merged_df['KIND_OF_HM_SET_ID']=='Left') & (merged_df['Orientation']=='EE')]
high_std_plot = sns.scatterplot(data=data_EE_Left[data_EE_Left['High_STD']], x='VTH_V', y='RSH_OHMSQR', color='green', label="EE_Left")
data_EE_Right = merged_df[(merged_df['KIND_OF_HM_SET_ID']=='Right') & (merged_df['Orientation']=='EE')]
high_std_plot = sns.scatterplot(data=data_EE_Right[data_EE_Right['High_STD']], x='VTH_V', y='RSH_OHMSQR', color='orange', label="EE_Right")


#high_std_plot = sns.scatterplot(data=data_WW_Right['High_STD']], x='VTH_V', y='RSH_OHMSQR', hue=['Orientation', 'KIND_OF_HM_SET_ID'], style='KIND_OF_HM_SET_ID', palette=custom_palette2,
#                markers=['o', 's'], hue_order=custom_hue_order, style_order=['Right', 'Left'],
#                hue_norm=None, sizes=None, size_order=None, size_norm=None,
#                alpha=0.7, edgecolor='none')
plt.title('High_STD')
plt.xlabel('VTH_V')
plt.ylabel('RSH_OHMSQR')
#plt.legend(title='Orientation', labels=[custom_labels[label] for label in ['WW_right', 'WW_left', 'EE_right', 'EE_left']])


# Plot for Low_STD
plt.subplot(1, 2, 2)
data_WW_Right = merged_df[(merged_df['KIND_OF_HM_SET_ID']=='Right') & (merged_df['Orientation']=='WW')]
low_std_plot = sns.scatterplot(data=data_WW_Right[data_WW_Right['Low_STD']], x='VTH_V', y='RSH_OHMSQR', color='red', label="WW_Right")
data_WW_Left = merged_df[(merged_df['KIND_OF_HM_SET_ID']=='Left') & (merged_df['Orientation']=='WW')]
low_std_plot = sns.scatterplot(data=data_WW_Left[data_WW_Left['Low_STD']], x='VTH_V', y='RSH_OHMSQR', color='blue',label="WW_Left")
data_EE_Left = merged_df[(merged_df['KIND_OF_HM_SET_ID']=='Left') & (merged_df['Orientation']=='EE')]
low_std_plot = sns.scatterplot(data=data_EE_Left[data_EE_Left['Low_STD']], x='VTH_V', y='RSH_OHMSQR', color='green', label="EE_Left")
data_EE_Right = merged_df[(merged_df['KIND_OF_HM_SET_ID']=='Right') & (merged_df['Orientation']=='EE')]
low_std_plot = sns.scatterplot(data=data_EE_Right[data_EE_Right['Low_STD']], x='VTH_V', y='RSH_OHMSQR', color='yellow', label="EE_Right")

#low_std_plot = sns.scatterplot(data=merged_df[merged_df['Low_STD']], x='VTH_V', y='RSH_OHMSQR', hue=['Orientation', 'KIND_OF_HM_SET_ID'], style='KIND_OF_HM_SET_ID', palette=custom_palette2,
#                markers=['o', 's'], hue_order=['WW', 'EE'], style_order=['Right', 'Left'],
#                hue_norm=None, sizes=None, size_order=None, size_norm=None,
#                alpha=0.7, edgecolor='none')
plt.title('Low_STD')
plt.xlabel('VTH_V')
plt.ylabel('RSH_OHMSQR')

print(merged_df[merged_df['Low_STD'==true]].count())
#plt.legend(title='Orientation', labels=[custom_labels[label] for label in ['WW_right', 'WW_left', 'EE_right', 'EE_left']])


# Create custom legend



handles, labels = high_std_plot.get_legend_handles_labels()
print(handles, labels)

#custom_handles = [handles[1], Line2D([], [], color='green', marker='s', linestyle='None'), handles[4], handles[5]]
#custom_labels = [labels[1], labels[2], labels[4], labels[5]]
#print(custom_handles)
#custom_legend_labels = ['{}_{}'.format(orientation, kind) for orientation in ['WW', 'EE'] for kind in ['Right', 'Left']]
##custom_legend = [(HandlerTuple(None), [(handle, label) for handle, label in zip(custom_handles, custom_legend_labels)])]
#print(custom_legend)


# Add legend
#plt.legend(title='Orientation_KIND_OF_HM_SET_ID', handles=custom_handles, labels=custom_legend_labels, handler_map={tuple: HandlerTuple(None)})



plt.tight_layout()
plt.show()