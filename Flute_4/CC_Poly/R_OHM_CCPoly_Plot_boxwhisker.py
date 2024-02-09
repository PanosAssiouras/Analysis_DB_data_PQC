import math
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
import pandas as pd
import pathlib
from scipy.optimize import curve_fit
from scipy.stats import norm
import seaborn as sns
import matplotlib.colors as mcolors

##Filechooser
root = tk.Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing
path=pathlib.Path().absolute()
#file_names = filedialog.askopenfilenames(initialdir=path, parent=root, title='Choose a file')

parameter = "R_OHM"
measurement = "CC_Poly"
axis_label = "Contact chain resitance [M$\Omega$]"
lower_limit = 0.0
upper_limit = 80

# 2.) Define fit function.
def fit_function(x, A, B, mu, sigma):
    #(A * np.exp(-x / beta) + B * np.exp(-1.0 * (x - mu) ** 2 / (2 * sigma ** 2)))
    return A+B*np.exp(-1.0 * (x - mu)**2 / (2 * sigma**2))


# ----------------------------------------------------------------------------------------------#
# ---------- Read data all and average .csv ----------------------------------------------------#
data = pd.read_csv("./" + parameter + "_all.csv", sep=",", skiprows=0)
Flute_results = pd.read_csv("./" + parameter + "_all.csv", sep=",", skiprows=0)
Flute_results_PSP = pd.read_csv("./" + parameter + "_all.csv", sep=",", skiprows=0)
Flute_average = pd.read_csv("./" + parameter + "_average_values.csv", sep=",", skiprows=0)
column_names = Flute_results.columns.values.tolist()
# ----------------------------------------------------------------------------------------------#

# ---------- Declare figure and axis parameters----------------------------------------------------#
fig, ax2 = plt.subplots()
xtick_labels = [name for name in Flute_average['batch_number']]
xtick_labels = xtick_labels[:(len(xtick_labels)-30)]
ax2.set_xticks(np.arange(len(xtick_labels)))
ax2.yaxis.set_minor_locator(MultipleLocator(10))
ax2.yaxis.set_major_locator(MultipleLocator(20))
ax2.tick_params(axis='x', which='major', labelsize=6, length=15)
ax2.tick_params(axis='x', which='minor', labelsize=6, length=5)
ax2.tick_params(axis='y', which='major', labelsize=20, length=15)
ax2.tick_params(axis='y', which='minor', labelsize=20, length=5)
ax2.set_xticklabels(xtick_labels, rotation=90)  # set axis rotation
ax2.set_xlim(lower_limit, len(xtick_labels))
ax2.yaxis.get_offset_text().set_fontsize(18)
# ----------------------------------------------------------------------------------------------#


names_of_2S = []
values_of_2S = []
errors_of_2S = []
names_of_PSs = []
values_of_PSs = []
errors_of_PSs = []
names_of_PSp = []
values_of_PSp = []
errors_of_PSp = []
values_of_2S_and_PSs = []
values_of_PSp_batches = []

#data = data.loc[(data[parameter] >= search_limits[param_number][0])
#                & (data[parameter] <= search_limits[param_number][1])]

for batch in xtick_labels:
    print(batch)
    values_of_2S = data.loc[ (data['Type']=="2-S") & (data['batch_number']==batch)][parameter]
    print(type(values_of_2S))
    print(values_of_2S)



# Set Seaborn style
#ax2 = sns.set(style="whitegrid")
median_color = 'yellow'
# Define custom colors for each 'Type'
mcolors.TABLEAU_COLORS
type_colors = {'PSS': "#2ecc71", '2S': "#3498db", 'PSP': '#FF4040'}  # Add more types and colors as needed
ax2.set_prop_cycle(color=['red','orange','yellow','green','blue','purple'])


# Map 'Type' to RGBA values
#data['flier_color'] = data['Type'].map(type_colors).map(mcolors.to_rgba)


# Create a bar and whisker plot using Seaborn with custom colors for each type
#plt.figure(figsize=(12, 8))
#ax2 = sns.boxplot(x='batch_number', y=parameter, hue='Type', data=data,
#                  palette=type_colors, medianprops=dict(color=median_color),
#                  flierprops=dict(markerfacecolor='gray', markersize=5, linestyle='none'), whis=(0, 100))
#ax2 = sns.boxplot(x='batch_number', y='S0_CMSEC', hue='Type', data=data.loc[ (data['Type']=="PSS")],
#                  palette=type_colors, medianprops=dict(color=median_color), flierprops=dict(markerfacecolor='red',
#                                                                                             markersize=5, linestyle='none'))
#ax2 = sns.boxplot(x='batch_number', y='S0_CMSEC', hue='Type', data=data.loc[ (data['Type']=="PSP")],
#                  palette=type_colors, medianprops=dict(color=median_color), flierprops=dict(markerfacecolor='red',
#                                                                                             markersize=5, linestyle='none'))
#plt.title('Batch Number vs. S0_CMSEC (Colored by Type)')
#plt.xlabel('Batch Number')
#plt.ylabel('S0_CMSEC')
#plt.legend(title='Type', bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.show()

for batch in xtick_labels:
    #type = data.loc[(data['batch_number']==batch)]['Type']
    #print(type)
    #if type == "2-S":
    ax2 = sns.boxplot(x='batch_number', y=parameter, hue='Type', data=data.loc[(data['batch_number']==batch) & (data['Type']=="2S")],
                      palette=type_colors, medianprops=dict(color=median_color),
                      flierprops=dict(markerfacecolor="#3498db", markersize=3, linestyle='none'))
    ax2 = sns.boxplot(x='batch_number', y=parameter, hue='Type', data=data.loc[(data['batch_number']==batch) & (data['Type']=="PSS")],
                      palette=type_colors, medianprops=dict(color=median_color),
                      flierprops=dict(markerfacecolor="#2ecc71", markersize=3, linestyle='none'))
    ax2 = sns.boxplot(x='batch_number', y=parameter, hue='Type', data=data.loc[(data['batch_number'] == batch) & (data['Type'] == "PSP")],
                      palette=type_colors, medianprops=dict(color=median_color),
                      flierprops=dict(markerfacecolor= '#FF4040', markersize=3, linestyle='none'))


# ---------- Make histogram inside the plot ----------------------------------------------------#
left, bottom, width, height = [0.20, 0.68, 0.25, 0.25]
ax3 = fig.add_axes([left, bottom, width, height])

values_of_2S_and_PSs = Flute_results[(Flute_results['Type'] == "PSS") | (Flute_results['Type'] == "2S")]
values_of_2S_and_PSs = values_of_2S_and_PSs[parameter]
values_of_2S_and_PSs_new = [value for value in values_of_2S_and_PSs if math.isnan(value) == False]

values_of_PSp_batches = Flute_results[Flute_results['Type'] == "PSP"]
values_of_PSp_batches = values_of_PSp_batches[parameter]
values_of_PSp_batches_new = [value for value in values_of_PSp_batches if math.isnan(value) == False]

# ----------------------------------------------------------------------------------------------------------------#
# ---------- Make histogram inside the plot ----------------------------------------------------#
# ---------- Calculate mean and std and make histo -----------------------------------------#
# ---------- 2S PSs ----------------------------------------------------#
(mu1, sigma1) = norm.fit(values_of_2S_and_PSs_new)
print("sigma1=", sigma1)
bins1 = np.linspace(lower_limit, upper_limit, 100)
data_entries_1, bins_1 = np.histogram(values_of_2S_and_PSs_new, bins=bins1)
binscenters1 = np.array([0.5 * (bins_1[i] + bins_1[i + 1]) for i in range(len(bins_1) - 1)])
popt, pcov = curve_fit(fit_function, xdata=binscenters1, ydata=data_entries_1, p0=[0.0, 0.0, mu1, sigma1])
n, bins, patches = ax3.hist(values_of_2S_and_PSs_new, color='cyan', alpha=1.0, bins=bins_1, label="2S and PSs wafers")
xspace = np.linspace(lower_limit, upper_limit, 10000)
sigma = popt[3]
onesigma = sigma * 2
# ax3.plot(xspace, fit_function(xspace, *popt), color='darkorange', alpha=0.8, linewidth=2.5, label=r'Fitted function') # for fitting
# ----------------------------------------------------------------------------------------------#
# ---------- Calculate mean and std and make histo -----------------------------------------#
# ---------- 2S PSs ----------------------------------------------------#
(mu2, sigma2) = norm.fit(values_of_PSp_batches_new)
bins2 = np.linspace(lower_limit, upper_limit, 100)
data_entries_2, bins_2 = np.histogram(values_of_PSp_batches_new, bins=bins2)
binscenters2 = np.array([0.5 * (bins_2[i] + bins_2[i + 1]) for i in range(len(bins_2) - 1)])
popt, pcov = curve_fit(fit_function, xdata=binscenters2, ydata=data_entries_2, p0=[0.0, 0.0, mu2, sigma2])
n, bins, patches = ax3.hist(values_of_PSp_batches_new, color='red', alpha=0.8, bins=bins2, label="PSp wafers")
xspace = np.linspace(lower_limit, upper_limit, 10000)
# ax3.plot(xspace, fit_function(xspace, *popt), color='darkorange', alpha=0.8, linewidth=2.5)
# ax3.set_xlim(left=(bin_left-2*(sign*bin_left/2)), right=(bin_right+2*(sign*bin_right/2)))
# ----------------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------------#
ax3.set_xlim(left=lower_limit, right=upper_limit)
ax3.set_ylim(bottom=0.0)
ax3.set_ylabel("Number of wafers", fontsize=18)
ax3.set_xlabel(axis_label, fontsize=18)
# set parameters for tick labels ##############################################
ax3.tick_params(axis='both', which='major', labelsize=15, length=10)
ax3.tick_params(axis='both', which='minor', labelsize=15, length=5)
ax3.legend(loc='best', prop={'size': 12})
ax3.xaxis.set_minor_locator(MultipleLocator(10))
ax3.xaxis.set_major_locator(MultipleLocator(20))
ax3.yaxis.set_minor_locator(MultipleLocator(50))
ax3.yaxis.set_major_locator(MultipleLocator(100))
ax3.xaxis.get_offset_text().set_fontsize(15)
# ----------------------------------------------------------------------------------------------#
# ---------- Straight line at mu and Spec limit ----------------------------------------------#
xmin, x_max = ax2.get_xlim()
x = np.linspace(xmin, x_max, 100)
print(xmin)
print(x_max)
ax2.set_title(measurement + " measurements", fontsize=20)
ax2.axhline(y=mu1, xmin=xmin, xmax=x_max, label="Mean value $\mu$", color="black")
# ax2.tick_params(axis='x', which='major', labelsize=4, length=10)
# ax2.axhline(y=-5, xmin=xmin, xmax=x_max, label="Spec limit", color="black", linestyle='--')
# ax2.axhline(y=2, xmin=xmin, xmax=x_max, label="Spec limit", color="black", linestyle='--')
# ----------------------------------------------------------------------------------------------#
# ----------Make color shaded area ----------------------------------------------#
# ax2.set_ylim(0.0, 4*round_up(max(data['Average']), 0))
# ax2.set_ylim(bottom=(bin_left - (sign * bin_left / 2)), top=(bin_right + (sign * bin_right / 2)))
ax2.set_ylim(bottom=lower_limit, top=upper_limit)
# ax2.set_ylim(0.0, 1.5 * round_up(max(data['Average']), 0))
ax2.fill_between(x, mu1 - abs(onesigma), mu1 + abs(onesigma), alpha=0.1,label="one sigma zone $(\mu-\sigma, \mu+\sigma)$", color="cyan")
ax2.set_xlabel("Batch number", fontsize=20)
ax2.set_ylabel(axis_label, fontsize=20)


# ax2.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
# ----------------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------------#
# ----------Force legend to show labels per type ----------------------------------------------#
handles, labels = ax2.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax2.legend(by_label.values(), by_label.keys())
# ax2.yaxis.set_minor_locator(MultipleLocator(1))
ax2.legend(by_label.values(), by_label.keys(), loc='upper right', prop={'size': 18})
# ----------------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------------#
#plt.savefig('.png', bbox_inches='tight')
width_screen = root.winfo_screenwidth()
height_screen = root.winfo_screenheight()
fig.set_size_inches(width_screen / 100, height_screen / 100)
fig.tight_layout()
fig.savefig("Bar_chart" + "_" + measurement +"_"+parameter+ ".pdf", dpi=100, bbox_inches='tight')
plt.show()
print("End")
# ----------------------------------------------------------------------------------------------#
