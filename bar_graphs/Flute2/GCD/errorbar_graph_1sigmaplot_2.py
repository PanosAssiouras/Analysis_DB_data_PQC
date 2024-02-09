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
    measurement=file.split(".")[0]
    print(measurement)

    if pos != -1:
        ##read data
        data = pd.read_csv(file_name, sep=",", skiprows=0)

        names_at_demok = []
        values_at_demok = []
        errors_at_demok = []
        names_at_per = []
        values_at_per = []
        errors_at_per = []
        names_at_brown = []
        values_at_brown = []
        errors_at_brown = []
        names_at_hephy = []
        values_at_hephy = []
        errors_at_hephy = []
        for index in data.index:
            print(data['Name'][index])
            if data['Center'][index] == "Demokritos":
                names_at_demok = [data['Name'][index]]
                values_at_demok = [data['Average'][index]]
                errors_at_demok = [data['Std.dev'][index]]
                ax2.errorbar(names_at_demok, values_at_demok, yerr=errors_at_demok, fmt='o', color='Blue', elinewidth=3,
                             capthick=3, errorevery=1, alpha=1, ms=4, capsize=5, label="Measured at Demokritos")
            elif data['Center'][index] == "Perugia":
                names_at_per=[data['Name'][index]]
                values_at_per=[data['Average'][index]]
                errors_at_per=[data['Std.dev'][index]]
                ax2.errorbar(names_at_per, values_at_per, yerr=errors_at_per, fmt='o', color='cyan', elinewidth=3,
                             capthick=3, errorevery=1, alpha=1, ms=4, capsize=5., label="Measured at INFN Perugia")
            elif data['Center'][index] == "Brown":
                names_at_brown = [data['Name'][index]]
                values_at_brown = [data['Average'][index]]
                errors_at_brown = [data['Std.dev'][index]]
                ax2.errorbar(names_at_brown, values_at_brown, yerr=errors_at_brown, fmt='o', color='red', elinewidth=3,
                             capthick=3, errorevery=1, alpha=1, ms=4, capsize=5, label="Measured at Brown")
            else:
                names_at_hephy = [data['Name'][index]]
                values_at_hephy = [data['Average'][index]]
                errors_at_hephy = [data['Std.dev'][index]]
                ax2.errorbar(names_at_hephy, values_at_hephy, yerr=errors_at_hephy, fmt='o', color='Green', elinewidth=3,
                             capthick=3, errorevery=1, alpha=1, ms=4, capsize=5, label="Measured at HEPHY")
        #############################################################################################################


        ######  Make histogram of average values #############################
        left, bottom, width, height = [0.20, 0.70, 0.25, 0.25]
        ax3 = fig.add_axes([left, bottom, width, height])
        #n, bins, patches = ax3.hist(data['Average'], range=(-6, 6), color='orange', bins=50, label="Histogram")
        #x_min, x_max = ax3.get_xlim()
        #x = np.linspace(min(data['Average'])-min(data['Average'])/2, max(data['Average'])+max(data['Average'])/2, 500)
        #y = norm.pdf(x, mu, sigma)
        #l = ax3.plot(x, y, 'r--', linewidth=2, label = "gauss fit")


        #### Calculate mean value and std.dev of data points ####################
        (mu, sigma) = norm.fit(data['Average'])
        #####################################################################
        bin_left = round_down(min(data['Average']), 0)
        bin_right = round_up(max(data['Average']), 0)
        print(bin_left, bin_right)
        if bin_left<0:
            sign=-1
        else:
            sign=1
        #bins = np.linspace(bin_left-(sign*bin_left/2), bin_right+(sign*bin_right/2), 100)
        bins = np.linspace(-1, 4, 50)
        print(bins)
        data_entries_1, bins_1 = np.histogram(data['Average'], bins=bins)
        binscenters = np.array([0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)])
        popt, pcov = curve_fit(fit_function, xdata=binscenters, ydata=data_entries_1, p0=[0.0, 0.0, mu, sigma])
        print(popt)
        n, bins, patches = ax3.hist(data['Average'], color='blue', alpha=0.5, bins=bins_1, label="Histogram")
        #ax3.bar(binscenters, data_entries_1, width=bins[1] - bins[0], color='navy', label=r'Histogram entries')
        #xspace = np.linspace((bin_left-(sign*bin_left/2)), (bin_right+(sign*bin_right/2)), 10000)
        #xspace = np.linspace((bin_left - (sign * bin_left / 2)), (bin_right + (sign * bin_right / 2)), 10000)
        xspace = np.linspace(-1, 4, 10000)
        ax3.plot(xspace, fit_function(xspace, *popt), color='darkorange', alpha=0.8, linewidth=2.5, label=r'Fitted function')
        #ax3.set_xlim(left=(bin_left-2*(sign*bin_left/2)), right=(bin_right+2*(sign*bin_right/2)))
        ax3.set_xlim(left=-1, right=4)
        ax3.set_ylim(bottom=0.0)
        ax3.set_ylabel("Number of batches", fontsize=15)
        ax3.set_xlabel("Average value of surface generation velocity [cm/s]", fontsize=15)
        # set parameters for tick labels ##############################################
        ax3.tick_params(axis='both', which='major', labelsize=15, length=10)
        ax3.tick_params(axis='both', which='minor', labelsize=15, length=5)
        ax3.legend(loc='upper right', prop={'size': 12})


        #### Straight line at mu value ######################################
        xmin, x_max = ax2.get_xlim()
        x = np.linspace(xmin, x_max, 100)
        print(xmin)
        print(x_max)
        ax2.axhline(y=mu, xmin=xmin, xmax=x_max, label="Mean value $\mu$", color="black")
        #####################################################################
        print("numb", 2*round_up(max(data['Average']), 0))
        ##### Make colored area betweewn mu-1sigma and mu+1sigma ########`
        #ax2.set_ylim(0.0, 4*round_up(max(data['Average']), 0))
        #ax2.set_ylim(bottom=(bin_left - (sign * bin_left / 2)), top=(bin_right + (sign * bin_right / 2)))
        ax2.set_ylim(bottom=-1, top=4)
        #ax2.set_ylim(0.0, 1.5 * round_up(max(data['Average']), 0))
        ax2.fill_between(x, popt[2] - popt[3], popt[2] + popt[3], alpha=0.1, label="one sigma zone $(\mu-\sigma, \mu+\sigma)$", color="gray")
        ax2.set_xlabel("Batch name", fontsize=25)
        ax2.set_ylabel("Surface generation velocity [cm/s]", fontsize=25)
        #####################################################################
        #### tick labels #############################################################
        xtick_labels = [ name for name in data['Name']]
        ax2.set_xticks(np.arange(len(xtick_labels)))
        # set parameters for tick labels ##############################################
        ax2.tick_params(axis='both', which='major', labelsize=20, length=10)
        ax2.tick_params(axis='both', which='minor', labelsize=20, length=5)
        ax2.set_xticklabels(xtick_labels, rotation=45)  # set axis rotation
        ax2.set_xlim(-1.0, len(xtick_labels))
        ##############################################################################


        ### force legend to show only one label per point ###########################
        handles, labels = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys())
        ax2.yaxis.set_minor_locator(MultipleLocator(1))
        ax2.legend(by_label.values(), by_label.keys(), loc='upper right', prop={'size': 18})
        ##################################################################



#plt.savefig('.png', bbox_inches='tight')
width_screen = root.winfo_screenwidth()
height_screen = root.winfo_screenheight()
fig.set_size_inches(width_screen / 100, height_screen / 100)
fig.tight_layout()
fig.savefig("Bar_chart" + "_" + measurement + ".pdf", dpi=100, bbox_inches='tight')
plt.show()

