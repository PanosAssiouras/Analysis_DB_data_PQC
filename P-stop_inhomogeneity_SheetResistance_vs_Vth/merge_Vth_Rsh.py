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



# Calculate standard deviation for each batch
std_dev_per_batch = merged_df.groupby('batch_number').agg({'VTH_V': 'std', 'RSH_OHMSQR': 'std'})

print(std_dev_per_batch)
print(max(std_dev_per_batch['VTH_V']), min(std_dev_per_batch['VTH_V']), std_dev_per_batch['VTH_V'].mean())


threshold = std_dev_per_batch.mean()

# Identify batches with very high deviation
batches_with_high_deviation = std_dev_per_batch[(std_dev_per_batch['VTH_V'] > threshold['VTH_V']) | (std_dev_per_batch['RSH_OHMSQR'] > threshold['RSH_OHMSQR'])]
print("Batches with very high deviation:")
print(batches_with_high_deviation)

merged_df['High_STD'] = ((merged_df['batch_number'].map(std_dev_per_batch['VTH_V']) > threshold['VTH_V']) &
                         (merged_df['batch_number'].map(std_dev_per_batch['RSH_OHMSQR']) > threshold['RSH_OHMSQR']))

merged_df['Low_STD'] = ((merged_df['batch_number'].map(std_dev_per_batch['VTH_V']) < threshold['VTH_V']) &
                         (merged_df['batch_number'].map(std_dev_per_batch['RSH_OHMSQR'])< threshold['RSH_OHMSQR']))


# Define custom labels for legend
custom_labels = {'WW_right': 'WW_right', 'WW_left': 'WW_left', 'EE_right': 'EE_right', 'EE_left': 'EE_left'}



# Create two plots for High_STD and Low_STD
plt.figure(figsize=(12, 6))


custom_palette = ["#FF5733", "#5733FF", "#FFFF33"]  # Example custom colors
sns.set_palette(custom_palette)
# Plot for High_STD
plt.subplot(1, 2, 1)
high_std_plot = sns.scatterplot(data=merged_df[merged_df['High_STD']], x='VTH_V', y='RSH_OHMSQR', hue='Orientation', style='KIND_OF_HM_SET_ID', palette=custom_palette,
                markers=['o', 's'], hue_order=['WW', 'EE'], style_order=['Right', 'Left'],
                hue_norm=None, sizes=None, size_order=None, size_norm=None,
                alpha=0.7, edgecolor='none')
plt.title('High_STD')
plt.xlabel('VTH_V')
plt.ylabel('RSH_OHMSQR')
#plt.legend(title='Orientation', labels=[custom_labels[label] for label in ['WW_right', 'WW_left', 'EE_right', 'EE_left']])


# Plot for Low_STD
plt.subplot(1, 2, 2)
low_std_plot = sns.scatterplot(data=merged_df[merged_df['Low_STD']], x='VTH_V', y='RSH_OHMSQR', hue='Orientation', style='KIND_OF_HM_SET_ID', palette=custom_palette,
                markers=['o', 's'], hue_order=['WW', 'EE'], style_order=['Right', 'Left'],
                hue_norm=None, sizes=None, size_order=None, size_norm=None,
                alpha=0.7, edgecolor='none')
plt.title('Low_STD')
plt.xlabel('VTH_V')
plt.ylabel('RSH_OHMSQR')
#plt.legend(title='Orientation', labels=[custom_labels[label] for label in ['WW_right', 'WW_left', 'EE_right', 'EE_left']])

'''
# Create custom legend

handles, labels = high_std_plot.get_legend_handles_labels()
print(handles, labels)

custom_handles = [handles[1], Line2D([], [], color='green', marker='s', linestyle='None'), handles[4], handles[5]]
custom_labels = [labels[1], labels[2], labels[4], labels[5]]
print(custom_handles)
custom_legend_labels = ['{}_{}'.format(orientation, kind) for orientation in ['WW', 'EE'] for kind in ['Right', 'Left']]
custom_legend = [(HandlerTuple(None), [(handle, label) for handle, label in zip(custom_handles, custom_legend_labels)])]
print(custom_legend)


# Add legend
plt.legend(title='Orientation_KIND_OF_HM_SET_ID', handles=custom_handles, labels=custom_legend_labels, handler_map={tuple: HandlerTuple(None)})

'''

plt.tight_layout()
plt.show()