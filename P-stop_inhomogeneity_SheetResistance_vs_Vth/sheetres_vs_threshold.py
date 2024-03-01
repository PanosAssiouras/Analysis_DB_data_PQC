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
merged_df.to_csv('merged_df.csv', index=False)
print(merged_df)

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

xaxis_label = 'Threshold voltage (V)'
yaxis_label = 'Sheet resistance (k$\Omega$/sq)'


data_WW_Right = merged_df[(merged_df['KIND_OF_HM_SET_ID']=='Right') & (merged_df['Orientation']=='WW')]
ax1.set_xlim(left=0.0, right=6.0)
ax1.set_ylim(bottom=16, top=30)
ax1.set_xlabel(xaxis_label, fontsize=12)
ax1.set_ylabel(yaxis_label, fontsize=12)
#ax1.tick_params(axis='x', which='major', labelsize=8, length=15)
#ax1.tick_params(axis='x', which='minor', labelsize=8, length=5)
#ax1.tick_params(axis='y', which='major', labelsize=8, length=15)
#ax1.tick_params(axis='y', which='minor', labelsize=8, length=5)
high_std_plot = sns.scatterplot(data=data_WW_Right, x='VTH_V', y='RSH_OHMSQR', color='red', label="WW_Right", ax=ax1)


data_WW_Left = merged_df[(merged_df['KIND_OF_HM_SET_ID']=='Left') & (merged_df['Orientation']=='WW')]
ax2.set_xlim(left=0.0, right=6.0)
ax2.set_ylim(bottom=16, top=30)
ax2.set_xlabel(xaxis_label, fontsize=12)
ax2.set_ylabel(yaxis_label, fontsize=12)
high_std_plot = sns.scatterplot(data=data_WW_Left, x='VTH_V', y='RSH_OHMSQR', color='blue', label="WW_Left", ax=ax2)


data_EE_Left = merged_df[(merged_df['KIND_OF_HM_SET_ID']=='Left') & (merged_df['Orientation']=='EE')]
ax3.set_xlim(left=0.0, right=6.0)
ax3.set_ylim(bottom=16, top=30)
ax3.set_xlabel(xaxis_label, fontsize=12)
ax3.set_ylabel(yaxis_label, fontsize=12)
high_std_plot = sns.scatterplot(data=data_EE_Left, x='VTH_V', y='RSH_OHMSQR', color='green', label="EE_Left", ax=ax3)


data_EE_Right = merged_df[(merged_df['KIND_OF_HM_SET_ID']=='Right') & (merged_df['Orientation']=='EE')]
ax4.set_xlim(left=0.0, right=6.0)
ax4.set_ylim(bottom=16, top=30)
ax4.set_xlabel(xaxis_label, fontsize=12)
ax4.set_ylabel(yaxis_label, fontsize=12)
high_std_plot = sns.scatterplot(data=data_EE_Right, x='VTH_V', y='RSH_OHMSQR', color='orange', label="EE_Right", ax=ax4)

#fig.tight_layout(pad=0.01)
#plt.subplots_adjust(left=0.1,
#                    bottom=0.1,
#                    right=0.9,
#                    top=0.9,
#                    wspace=0.4,
#                    hspace=0.4)
#plt.show()

fig1.tight_layout()
fig1.savefig("Rsh_Vth_WW_Right"+".pdf", dpi=200, bbox_inches='tight')

fig2.tight_layout()
fig2.savefig("Rsh_Vth_WW_Left"+".pdf", dpi=200, bbox_inches='tight')

fig3.tight_layout()
fig3.savefig("Rsh_Vth_EE_Left"+".pdf", dpi=200, bbox_inches='tight')

fig4.tight_layout()
fig4.savefig("Rsh_Vth_EE_Right"+".pdf", dpi=200, bbox_inches='tight')
plt.show()
print("End")