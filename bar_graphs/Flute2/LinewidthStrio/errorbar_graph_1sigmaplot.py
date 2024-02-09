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
from scipy import optimize
from scipy.optimize import fsolve
from scipy.stats import norm


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

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
            numberOfdigits -= 1
        print(number)
    elif number<1.0:
        numberOfdigits = 0
        while (number < 1.0):
            number = number * 10
            numberOfdigits += 1
        print(number)
    else:
        numberOfdigits+=3
    return numberOfdigits

fig, ax2 = plt.subplots()
fig.suptitle('Line width of n+ implant', fontsize=20)
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

        #### tick labels #############################################################
        xtick_labels = [ name for name in data['Name']]
        ax2.set_xticks(np.arange(len(xtick_labels)))
        # set parameters for tick labels ##############################################
        ax2.tick_params(axis='both', which='major', labelsize=20, length=10)
        ax2.tick_params(axis='both', which='minor', labelsize=20, length=5)
        ax2.set_xticklabels(xtick_labels, rotation=45)  # set axis rotation
        ax2.set_xlim(-1.0, len(xtick_labels))
        ##############################################################################

        #### Calculate mean value and std.dev of data points ####################
        (mu, sigma) = norm.fit(data['Average'])
        #####################################################################

        #### Straight line at mu value ######################################
        xmin, x_max = ax2.get_xlim()
        x = np.linspace(xmin, x_max, 100)
        print(xmin)
        print(x_max)
        ax2.axhline(y=mu, xmin=xmin, xmax=x_max, label="Mean value $\mu$", color="black")
        #####################################################################

        ##### Make colored area betweewn mu-1sigma and mu+1sigma ########
        ymax = max(data["Average"])
        ymin = min(data["Average"])
        n = number_of_digits_to_round(ymax)
        ax2.set_ylim(0.0, 30+round_up(ymax, n))
        ax2.fill_between(x, mu - sigma, mu + sigma, alpha=0.1, label="one sigma zone $(\mu-\sigma, \mu+\sigma)$", color="gray")
        ax2.set_xlabel("Batch name", fontsize=25)
        ax2.set_ylabel("Line width [\u03bc m]", fontsize=20)
        #####################################################################

        ### force legend to show only one label per point ###########################
        handles, labels = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys())
        ax2.yaxis.set_minor_locator(MultipleLocator(1))
        ax2.legend(by_label.values(), by_label.keys(), loc='upper right', prop={'size': 18})
        ##################################################################

        ######  Make histogram of average values #############################
        left, bottom, width, height = [0.20, 0.65, 0.25, 0.25]
        ax3 = fig.add_axes([left, bottom, width, height])
        n, bins, patches = ax3.hist(data['Average'], range=(-(10+round_up(ymax, n)), 10+round_up(ymax, n)), color='orange', bins=150, label="Histogram")
        x_min, x_max = ax3.get_xlim()
        x = np.linspace(x_min, x_max, 200)
        y = norm.pdf(x, mu, sigma)
        l = ax3.plot(x, y, 'r--', linewidth=2, label = "gauss fit")
        ax3.set_ylabel("Number of batches", fontsize=15)
        ax3.set_xlabel("Line width [\u03bc m]", fontsize=12)
        # set parameters for tick labels ##############################################
        ax3.tick_params(axis='both', which='major', labelsize=15, length=10)
        ax3.tick_params(axis='both', which='minor', labelsize=15, length=5)
        ax3.legend(loc='best', prop={'size': 15})


#plt.savefig('.png', bbox_inches='tight')
width_screen = root.winfo_screenwidth()
height_screen = root.winfo_screenheight()
fig.set_size_inches(width_screen / 100, height_screen / 100)
fig.tight_layout()
fig.savefig("Bar_chart" + "_" + measurement + ".pdf", dpi=100, bbox_inches='tight')
plt.show()

