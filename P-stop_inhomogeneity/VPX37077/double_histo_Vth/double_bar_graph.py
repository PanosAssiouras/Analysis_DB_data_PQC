import math
import os
import tkinter as tk
from tkinter import filedialog
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
from scipy import optimize
from scipy.optimize import fsolve

# convert list to dictionary
def convert(lst):
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
    return res_dct

# Denoting fitting function
def fit_func(x,a, b):
    return a * x + b

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
        numberOfdigits+=1
    return numberOfdigits
#def findIntersection(fun1,fun2,x0):
#     return fsolve(lambda x :fun1(x) - fun2(x),x0)

##Filechooser
fig2, ax2 = plt.subplots()


root = tk.Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing
path=pathlib.Path().absolute()
file_names = filedialog.askopenfilenames(initialdir=path, parent=root, title='Choose a file')
##Loop in each file
for file_name in file_names:
    pos=file_name.find(".csv")
    #pos=file_name.find(".csv")
    file= file_name.split("/")[-1]
    measurment=file.split(".")[0]
    print(measurment)
    if (pos!=-1):
        ##read data
        data=pd.read_csv(file_name, sep=",", skiprows=0)
        print(data)
        file_name = os.path.splitext(file_name)[0]

        width=0.3
        print(data["Name"])
        xtick_labels = []
       # for name in data['Name']:
            #xtick_labels =  convert
            #xtick_labels.append(name.split("_")[1]+"_"+name.split("_")[2]+"_"+name.split("_")[5])

#            print(tick_name)
#            xtick_labels = convert(tick_name)

        xtick_labels = convert([name.split("_")[1]+"_"+name.split("_")[2]+"_"+name.split("_")[5] for name in data['Name']])
        type(xtick_labels)
        print(xtick_labels)
        #ind = np.arange(len(xtick_labels))+1
        ind = np.arange(len(xtick_labels))
        print("ind=", ind)
        ax2.set_xticks(ind + width / 2)
        ax2.set_xticklabels(xtick_labels)
        #ax2.set_xticks(ind+0.5)
        #east_left_values = []
        #west_left_values = []
        #east_right_values = []
        #west_right_values = []
        east_values = []
        west_values = []
        for index, row in data.iterrows():
            name=row[0]
            print(name)
            if name.find("E") != -1:
                east_values.append(row[1])
            elif name.find("W") != -1:
                west_values.append(row[1])
        print(east_values)
        print(west_values)
        for tick in ind:
            fig2, ax2.bar(ind[tick], east_values[tick], alpha=0.7, width=0.3, ecolor='black', capsize=10, color='blue')
            fig2, ax2.bar(ind[tick]+width, west_values[tick], alpha=0.7, width=0.3, ecolor='black', capsize=10, color='cyan')

        colors = {'East halfmoon': 'blue', 'West halfmoon' : 'cyan'}
        labels = list(colors.keys())
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
        ax2.set_ylim(1.0, 6.0)
        ax2.set_title("Threshold voltage (VPX37077)", fontsize=35)
        plt.legend(handles, labels, prop={'size': 30})
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.xlabel('Wafer serial number',fontsize=38)
        plt.ylabel('Threshold voltage [V]', fontsize=38)
        plt.xticks(rotation=40)

        #ax2.legend((rects1[0], rects2[0]), ('Men', 'Women'))
        ax2.set_xlim(-1.0, len(xtick_labels))
        ax2.tick_params(axis='both', which='minor', labelsize=30, length=20)
        ax2.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax2.yaxis.set_major_locator(MultipleLocator(1.0))
        # Add text near the vertical line
        ax2.axvline(x=6.5, color='red', linestyle='--', linewidth=2)
        ax2.text(8.0, 4.8, 'Right flute set', color='red', ha='center', fontsize=35)
        ax2.text(3.5, 4.8, 'Left flute set', color='red', ha='center', fontsize=35)

        # horizontal line indicating the threshold
        x_min, x_max = ax2.get_xlim()
       # ax2.plot([x_min, x_max], [3.5, 3.5], "k--", color='gray')
       # ax2.plot([-0.5, 16.5], [20.0, 20.0], "k--", color='gray')

#plt.savefig('.png', bbox_inches='tight')
width = root.winfo_screenwidth()
height = root.winfo_screenheight()
fig2.set_size_inches(width / 100, height / 50)
fig2.savefig("Bar_chart" + "_" + measurment + ".pdf", dpi=100, bbox_inches='tight')


plt.show()

