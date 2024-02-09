import math
import os
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilenames
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
import pandas as pd
import pathlib
from scipy.optimize import curve_fit
from scipy import optimize
from scipy.optimize import fsolve
from scipy.stats import norm


parameter = "|NOX|"
measurement = "MOS_QUARTER"

def count_digits(n):
    count = 0
    while (n > 0):
        count = count + 1
        n = n // 10
    return  count

def round_up(n, decimals=0):
    if n.is_integer():
        number_of_digits = count_digits(n)
        n = n/10**(number_of_digits-decimals)
        multiplier = 10 ** decimals
        round_n = math.ceil(n * multiplier) / multiplier
    else:
        multiplier = 10 ** decimals
        round_n = math.ceil(n * multiplier) / multiplier
    return round_n

def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier


def number_of_digits_to_round(number):
    number=abs(number)
    numberOfdigits = 0
    if number > 10:
        numberOfdigits = 0
        while (number > 10):
            number = number /10
            numberOfdigits += 1
        print(number)
    elif number<1.0:
        numberOfdigits = 0
        while (number < 1.0):
            number = number * 10
            numberOfdigits += 1
        print(number)
    else:
        numberOfdigits+=1
    return numberOfdigits
#def findIntersection(fun1,fun2,x0):
#     return fsolve(lambda x :fun1(x) - fun2(x),x0)

# 2.) Define fit function.
def fit_function(x, A, B, mu, sigma):
    #(A * np.exp(-x / beta) + B * np.exp(-1.0 * (x - mu) ** 2 / (2 * sigma ** 2)))
    return A+B*np.exp(-1.0 * (x - mu)**2 / (2 * sigma**2))

fig, ax2 = plt.subplots()

##Filechooser
root = tk.Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing
path=pathlib.Path().absolute()
file_names = filedialog.askopenfilenames(initialdir=path, parent=root, title='Choose a file')

##Loop in each file
for file_name in file_names:
    pos = file_name.find(".csv")
    file = file_name.split("/")[-1]

    if pos != -1:
        ##read data
        data = pd.read_csv(file_name, sep=",", skiprows=0)

        Flute1_results = pd.read_csv("./MOS_NOX_all.csv", sep=",", skiprows=0)
        Flute1_results_PSP = pd.read_csv("./MOS_NOX_all.csv", sep=",", skiprows=0)

        column_names = Flute1_results.columns.values.tolist()
        #measurement = file.split(".")[0]
        #print(measurement)

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
        for index in data.index:
            print(data['Name'][index])
            print(type(data['Name'][index]))
            if data['Type'][index] == "2-S":
                names_of_2S = [str(data['Name'][index])]
                values_of_2S = [abs((data['Average'][index]))]
                errors_of_2S = [(data['Std.dev'][index])]
               # values_of_2S_and_PSs.append((data['Average'][index]))
                ax2.errorbar(names_of_2S, values_of_2S, yerr=errors_of_2S, fmt='o', color='Blue', elinewidth=1,
                             capthick=1, errorevery=1, alpha=1, ms=5, capsize=4, label="2S batches")
            elif data['Type'][index] == "PSS":
                names_of_PSs=[str(data['Name'][index])]
                values_of_PSs=[abs(data['Average'][index])]
                errors_of_PSs=[(data['Std.dev'][index])]
               # values_of_2S_and_PSs.append((data['Average'][index]))
                ax2.errorbar(names_of_PSs, values_of_PSs, yerr=errors_of_PSs, fmt='^', color='Green', elinewidth=1,
                             capthick=1, errorevery=1, alpha=1, ms=5, capsize=4., label="PS-s batches")
            elif data['Type'][index] == "PSP":
                names_of_PSp = [str(data['Name'][index])]
                values_of_PSp = [abs(data['Average'][index])]
                errors_of_PSp = [(data['Std.dev'][index])]
               # values_of_PSp_batches.append((data['Average'][index]))
                ax2.errorbar(names_of_PSp, values_of_PSp, yerr=errors_of_PSp, fmt='s', color='red', elinewidth=1,
                             capthick=1, errorevery=1, alpha=1, ms=5, capsize=4, label="PS-p batches")
        #############################################################################################################



        ######  Make histogram of average values #############################
        left, bottom, width, height = [0.20, 0.68, 0.25, 0.25]
        ax3 = fig.add_axes([left, bottom, width, height])
        #n, bins, patches = ax3.hist(data['Average'], range=(-6, 6), color='orange', bins=50, label="Histogram")
        #x_min, x_max = ax3.get_xlim()
        #x = np.linspace(min(data['Average'])-min(data['Average'])/2, max(data['Average'])+max(data['Average'])/2, 500)
        #y = norm.pdf(x, mu, sigma)
        #l = ax3.plot(x, y, 'r--', linewidth=2, label = "gauss fit")

        values_of_2S_and_PSs = Flute1_results[(Flute1_results['Type'] == "PSS") | (Flute1_results['Type'] == "2-S")]
        values_of_2S_and_PSs = values_of_2S_and_PSs[parameter]
        values_of_2S_and_PSs_new = [abs(value) for value in values_of_2S_and_PSs if math.isnan(value) == False]

        #values_of_PSp_batches = Flute1_results_PSP['|NOX|']
        #values_of_PSp_batches_new = [value for value in values_of_PSp_batches if math.isnan(value) == False]

        values_of_PSp_batches = Flute1_results[Flute1_results['Type'] == "PSP"]
        values_of_PSp_batches = values_of_PSp_batches[parameter]
        values_of_PSp_batches_new = [abs(value) for value in values_of_PSp_batches if math.isnan(value) == False]

        #####################################################################
        bin_left = round_down(min(data['Average']), 0)
        bin_right = round_up(max(data['Average']), 0)
        print(bin_left, bin_right)
        if bin_left<0:
            sign=-1
        else:
            sign=1
        #2S and PSs batches
        #### Calculate mean value and std.dev of data points ####################4

        (mu1, sigma1) = norm.fit(values_of_2S_and_PSs_new)
        bins1 = np.linspace(0.0, 30E+10, 100)
        data_entries_1, bins_1 = np.histogram(values_of_2S_and_PSs_new, bins=bins1)
        binscenters1 = np.array([0.5 * (bins_1[i] + bins_1[i + 1]) for i in range(len(bins_1) - 1)])
        popt, pcov = curve_fit(fit_function, xdata=binscenters1, ydata=data_entries_1, p0=[0.0, 0.0, mu1, sigma1])
        n, bins, patches = ax3.hist(values_of_2S_and_PSs_new, color='cyan', alpha=1.0, bins=bins_1, label="2S and PSs wafers")
        xspace = np.linspace(0.0, 30E+10, 10000)
        #ax3.plot(xspace, fit_function(xspace, *popt), color='darkorange', alpha=0.8, linewidth=2.5, label=r'Fitted function')


        #PSp batches
        #### Calculate mean value and std.dev of data points ####################4
        (mu2, sigma2) = norm.fit(values_of_PSp_batches_new)
        bins2 = np.linspace(0, 30E+10, 100)
        data_entries_2, bins_2 = np.histogram(values_of_PSp_batches_new, bins=bins2)
        binscenters2 = np.array([0.5 * (bins_2[i] + bins_2[i + 1]) for i in range(len(bins_2) - 1)])
        popt, pcov = curve_fit(fit_function, xdata=binscenters2, ydata=data_entries_2, p0=[0.0, 0.0, mu2, sigma2])
        n, bins, patches = ax3.hist(values_of_PSp_batches_new, color='red', alpha=0.5, bins=bins2, label="PSp wafers")
        #xspace = np.linspace(0.0, 20.0, 10000)
        #ax3.plot(xspace, fit_function(xspace, *popt), color='darkorange', alpha=0.8, linewidth=2.5)
        #ax3.set_xlim(left=(bin_left-2*(sign*bin_left/2)), right=(bin_right+2*(sign*bin_right/2)))

        ax3.set_xlim(left=0, right=30E+10)
        ax3.set_ylim(bottom=0.0)
        ax3.set_ylabel("Number of wafers", fontsize=18)
        ax3.set_xlabel("NOX [cm$^{-2}$]", fontsize=18)
        # set parameters for tick labels ##############################################
        ax3.tick_params(axis='both', which='major', labelsize=15, length=10)
        ax3.tick_params(axis='both', which='minor', labelsize=15, length=5)
        ax3.legend(loc='best', prop={'size': 12})
        ax3.xaxis.set_minor_locator(MultipleLocator(2.5E+10))
        ax3.xaxis.set_major_locator(MultipleLocator(5.0E+10))
        ax3.yaxis.set_minor_locator(MultipleLocator(50))
        ax3.yaxis.set_major_locator(MultipleLocator(100))
        ax3.xaxis.get_offset_text().set_fontsize(15)


        #### Straight line at mu value ######################################
        xmin, x_max = ax2.get_xlim()
        x = np.linspace(xmin, x_max, 100)
        print(xmin)
        print(x_max)
        ax2.set_title(measurement + " measurements", fontsize=20)
        ax2.axhline(y=mu1, xmin=xmin, xmax=x_max, label="Mean value $\mu$", color="black")
        #ax2.axhline(y=2, xmin=xmin, xmax=x_max, label="Spec limit", color="black", linestyle='--')
        #####################################################################
        print("numb", 2*round_up(max(data['Average']), 0))
        ##### Make colored area betweewn mu-1sigma and mu+1sigma ########`
        #ax2.set_ylim(0.0, 4*round_up(max(data['Average']), 0))
        #ax2.set_ylim(bottom=(bin_left - (sign * bin_left / 2)), top=(bin_right + (sign * bin_right / 2)))
        ax2.set_ylim(bottom=0, top=30E+10)
        #ax2.set_ylim(0.0, 1.5 * round_up(max(data['Average']), 0))
        ax2.fill_between(x, mu1 - sigma1, mu1 + sigma1, alpha=0.1, label="one sigma zone $(\mu-\sigma, \mu+\sigma)$", color="cyan")
        ax2.set_xlabel("Batch name", fontsize=20)
        ax2.set_ylabel("NOX [cm$^{-2}$]", fontsize=20)
        ax2.yaxis.set_minor_locator(MultipleLocator(1.0E+10))
        #####################################################################
        #### tick labels #############################################################
        xtick_labels = [ name for name in data['Name']]
        ax2.set_xticks(np.arange(len(xtick_labels)))
        # set parameters for tick labels ##############################################
        ax2.yaxis.set_minor_locator(MultipleLocator(2.5E+10))
        ax2.yaxis.set_major_locator(MultipleLocator(5E+10))
        ax2.tick_params(axis='x', which='major', labelsize=10, length=15)
        ax2.tick_params(axis='x', which='minor', labelsize=10, length=5)
        ax2.tick_params(axis='y', which='major', labelsize=20, length=15)
        ax2.tick_params(axis='y', which='minor', labelsize=20, length=5)
        ax2.set_xticklabels(xtick_labels, rotation=90)  # set axis rotation
        ax2.set_xlim(-1.0, len(xtick_labels))
        ax2.yaxis.get_offset_text().set_fontsize(18)
        #ax2.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ##############################################################################


        ### force legend to show only one label per point ###########################
        handles, labels = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys())
        #ax2.yaxis.set_minor_locator(MultipleLocator(1))
        ax2.legend(by_label.values(), by_label.keys(), loc='upper right', prop={'size': 18})
        ##################################################################



#plt.savefig('.png', bbox_inches='tight')
width_screen = root.winfo_screenwidth()
height_screen = root.winfo_screenheight()
fig.set_size_inches(width_screen / 100, height_screen / 100)
fig.tight_layout()
fig.savefig("Bar_chart" + "_" + measurement +"_"+parameter+ ".pdf", dpi=100, bbox_inches='tight')
plt.show()

