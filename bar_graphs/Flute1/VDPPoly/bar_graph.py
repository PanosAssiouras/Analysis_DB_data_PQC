import math
import os
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilenames
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
from scipy import optimize
from scipy.optimize import fsolve

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
file_names = filedialog.askopenfilenames(initialdir=path,parent=root,title='Choose a file')
##Loop in each file
for file_name in file_names:
    pos=file_name.find(".csv")
    #pos=file_name.find(".csv")
    file= file_name.split("/")[-1]
    measurment=file.split(".")[0]
    print(measurment)
    if (pos!=-1):
        ##read data
        data=pd.read_csv(file_name, sep=",",skiprows=0)
        print(data)
        file_name = os.path.splitext(file_name)[0]
        #names = data.iloc[:,0]
        average = data.iloc[:,3]

        max_value = max(average)
        min_value = min(average)

        for index, row in data.iterrows():
            name=row[0]
            if row[1]=="Demokritos":
                fig2, ax2.bar(row[0], row[2],yerr=row[3], alpha=0.5, ecolor='black', capsize=10, label='Measured at Demokritos', color='blue')
            elif row[1]=="Perugia":
                fig2, ax2.bar(row[0], row[2], yerr=row[3], alpha=0.5, ecolor='black', capsize=10, label='Measured at Perugia', color='cyan')
            elif row[1]=="HEPHY":
                fig2, ax2.bar(row[0], row[2], yerr=row[3], alpha=0.5, ecolor='black', capsize=10, label='Measured at HEPHY', color='green')
            else:
                fig2, ax2.bar(row[0], row[2], yerr=row[3], alpha=0.5, ecolor='black', capsize=10, label="Measured at Brown", color='red')

        colors = {'Measured at Demokritos': 'blue', 'Measured at Perugia': 'cyan', 'Measured at Brown': 'red',
                  'Measured at HEPHY': 'green'}
        labels = list(colors.keys())
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
        digits = int(number_of_digits_to_round(max_value))
        ax2.set_ylim(2.0,2.8)
        ax2.set_title(measurment, fontsize=20)
        plt.legend(handles, labels, prop={'size': 20})
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.xlabel('Batch-number',fontsize=20)
        plt.ylabel('Sheet Resistance [k\u03A9/square]', fontsize=20)
        plt.xticks(rotation=45)

        # horizontal line indicating the threshold
        ax2.plot([-0.5,17.5], [2.2, 2.2], "k--" , color='gray')
        ax2.plot([-0.5, 17.5], [2.4, 2.4], "k--",  color='gray')


#plt.savefig('.png', bbox_inches='tight')
width = root.winfo_screenwidth()
height = root.winfo_screenheight()
fig2.set_size_inches(width / 100, height / 100)
fig2.savefig("Bar_chart" + "_" + measurment + ".png", dpi=100, bbox_inches='tight')

plt.show()

