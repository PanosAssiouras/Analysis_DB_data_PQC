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

