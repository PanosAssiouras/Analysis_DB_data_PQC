import math
import os
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilenames
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pathlib
from scipy import optimize
from scipy.optimize import fsolve
from scipy.stats import norm

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
fig, ax2= plt.subplots( figsize=(20, 10))

root = tk.Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing
path=pathlib.Path().absolute()
file_names = filedialog.askopenfilenames(initialdir = path, parent = root, title = 'Choose a file')
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

        # now, define the ticks (i.e. locations where the labels will be plotted)
        #xticks = [i for i in range(len(data.index))]

        #print(xticks)

        # also define the labels we'll use (note this MUST have the same size as `xticks`!)
        #xtick_labels = [ name for name in data['Name']]
        #values = [value for value in data['Average']]
        #print(values)
        #print(xtick_labels)
        #ax2.hist(values, bins= len(data.index), align='left')  # `align='left'` is used to center the labels
        #ax2.set_xticks(xticks)
        #ax2.set_xticklabels(xtick_labels)


        #ax2.errorbar(data['Name'], data['Average'], yerr=data['Std.dev'], fmt='o', color='Blue', elinewidth=3,
        #             capthick=3,
        #             errorevery=1, alpha=1, ms=4, capsize=5)


        names_at_demok = []
        values_at_demok = []
        erros_at_demok = []
        names_at_per = []
        values_at_per = []
        erros_at_per = []
        names_at_brown = []
        values_at_brown = []
        erros_at_brown = []
        names_at_hephy = []
        values_at_hephy = []
        erros_at_hephy = []
        colors = []
        labels = []
        for index in data.index:
            #names=[row[0]]
            #print(row[0], row[2])
            print(data['Name'][index])
            if data['Center'][index] == "Demokritos":
                # names_at_demok.append(data['Name'][index])
                # values_at_demok.append(data['Average'][index])
                # erros_at_demok.append(data['Std.dev'][index])
                names_at_demok = [data['Name'][index]]
                values_at_demok = [data['Average'][index]]
                erros_at_demok = [data['Std.dev'][index]]
                colors.append('Blue')
                labels.append("Measured at Demokritos")
            elif data['Center'][index] == "Perugia":
                # names_at_per.append(data['Name'][index])
                # values_at_per.append(data['Average'][index])
                # erros_at_per.append(data['Std.dev'][index])
                names_at_per=[data['Name'][index]]
                values_at_per=[data['Average'][index]]
                erros_at_per=[data['Std.dev'][index]]
                colors.append('Cyan')
                labels.append("Measured at INFN Perugia")
            elif data['Center'][index] == "Brown":
                # names_at_brown.append(data['Name'][index])
                # values_at_brown.append(data['Average'][index])
                # erros_at_brown.append(data['Std.dev'][index]

                names_at_brown = [data['Name'][index]]
                values_at_brown = [data['Average'][index]]
                erros_at_brown = [data['Std.dev'][index]]

                colors.append('Red')
                labels.append("Measured at Brown")
            else:
                names_at_hephy = [data['Name'][index]]
                values_at_hephy = [data['Average'][index]]
                erros_at_hephy = [data['Std.dev'][index]]

                colors.append('Green')

            print(names_at_demok)
            ax2.errorbar(names_at_demok, values_at_demok, yerr=erros_at_demok, fmt='o', color='Blue', elinewidth=3,
                 capthick=3, errorevery=1, alpha=1, ms=4, capsize=5, label="Measured at Demokritos")
            ax2.errorbar(names_at_per, values_at_per, yerr=erros_at_per, fmt='o', color='cyan', elinewidth=3,
                 capthick=3, errorevery=1, alpha=1, ms=4, capsize=5., label="Measured at INFN Perugia")
            ax2.errorbar(names_at_brown, values_at_brown, yerr=erros_at_brown, fmt='o', color='red', elinewidth=3,
                 capthick=3, errorevery=1, alpha=1, ms=4, capsize=5, label="Measured at Brown")
            ax2.errorbar(names_at_hephy, values_at_hephy, yerr=erros_at_hephy, fmt='o', color='Green', elinewidth=3,
                         capthick=3, errorevery=1, alpha=1, ms=4, capsize=5, label="Measured at HEPHY")
        #ax2.legend(loc='upper left')

        #colors = {'Measured at Demokritos': 'blue', 'Measured at Perugia': 'cyan', 'Measured at Brown': 'red',
        #          'Measured at HEPHY': 'green'}

        ##### Removes duplicate values from legend  #######################
        #ax2.rcParams["figure.figsize"] = [7.00, 3.50]
        #ax2.rcParams["figure.autolayout"] = True

        handles, labels = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        #ax2.set_xticklabels(rotation=(45), fontsize=10)
        xtick_labels = [ name for name in data['Name']]
        values = [value for value in data['Average']]
        print(values)
        print(len(xtick_labels))
        #ax2.hist(values, bins= len(data.index), align='left')  # `align='left'` is used to center the labels
        ax2.set_xticks(np.arange(len(xtick_labels)))
        # set parameters for tick labels
        ax2.tick_params(axis='both', which='major', labelsize=10, length=10)
        ax2.tick_params(axis='both', which='minor', labelsize=10, length=5)
        #ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax2.set_xticklabels(xtick_labels, rotation=(45), fontsize=15)
        ax2.set_xlim(-1.0, len(xtick_labels))
        ax2.legend(by_label.values(), by_label.keys())
        fig.tight_layout()
        ##################################################################


        (mu, sigma) = norm.fit(data['Average'])
        #####################################################################

        #### Straight line at mu value ######################################

        #####################################################################
        ##### Make colored area betweewn mu-1sigma and mu+1sigma ########
        xmin, xmax = ax2.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        print(xmin)
        print(xmax)
        #ax2.axline([xmin+5.0, mu], [xmax-5.0, mu], color='black', label = 'mean value')
        ax2.axhline(y = mu, xmin = xmin,  xmax = xmax, label="Mean value")
        ax2.set_ylim(0.0, -6.0)
        ax2.fill_between(x, mu - sigma, mu + sigma, alpha=0.1, label="1-Sigma")
        ax2.set_xlabel("Batch name", fontsize=20)
        ax2.set_ylabel("Flatband Voltage [V]", fontsize=20)

        handles, labels = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys(), loc='upper right', prop={'size': 18})
        ##################################################################


        ######  Make histogram of average values #############################
        left, bottom, width, height = [0.10, 0.65, 0.3, 0.3]
        ax3 = fig.add_axes([left, bottom, width, height])
        n, bins, patches = ax3.hist(data['Average'], range=(-10, 10), color='orange', bins=50, label="Histogram")
        # add a 'best fit' line
        xmin, xmax = ax3.get_xlim()
        x = np.linspace(xmin, xmax, 300)
        y = norm.pdf(x, mu, sigma)
        l = ax3.plot(x, y, 'r--', linewidth=2, label = "gaus fit")
        ax3.set_ylabel("Number of batches")
        ax3.set_xlabel("Average value of flat-band voltage")
        ax3.legend(loc='upper right', prop={'size': 10})

        ### Set axis ticks and titles ##################################
        ##=ax2.set_xticks(fontsize=13)
        #ax2.set_yticks(fontsize=13)
        #ax2.xlabel('Batch-number', fontsize=20)
        #ax2.ylabel('Threshold Voltage [V]', fontsize=20)
        #ax2.xticks(rotation=45)
        ###############################################################
        #print(handles)

        #labels = list(colors.keys())
        #handles = [plt.Circle((0, 0), 1, color=colors[label]) for label in labels]
        #handles, labels = plt.gca().get_legend_handles_labels()
        #by_label = dict(zip(labels, handles))
        #digits = int(number_of_digits_to_round(max_value))
        #ax2.set_ylim(2.0, 6.0)
        #ax2.set_title(measurment, fontsize=20)
        #plt.legend(handles, labels, prop={'size': 20})
        #ax2.legend(handles, labels, prop={'size': 20})
        #ax2.legend(loc='upper left')
            #fig2, ax2.errorbar(data['Name'][index], data['Average'][index], yerr=data['Std.dev'][index], fmt='o', color='Blue', elinewidth=3,
            #                  capthick=3, errorevery=1, alpha=1, ms=4, capsize=5)
        #    if row[1]=="Demokritos":
        #        #fig2, ax2.bar(row[0], row[2],yerr=row[3], alpha=0.5, ecolor='black', capsize=10, label='Measured at Demokritos', color='blue')

        #        fig2, ax2.errorbar(50, row[2], yerr=row[3], fmt='o', color='Blue',elinewidth=3,
        #                          capthick=3, errorevery=1, alpha=1, ms=4, capsize=5)
            #elif row[1]=="Perugia":
                #fig2, ax2.bar(row[0], row[2], yerr=row[3], alpha=0.5, ecolor='black', capsize=10, label='Measured at Perugia', color='cyan')
            #    fig2, ax2.errorbar(row[0], row[2], yerr=row[3], fmt='o', color='Blue',
             #                      elinewidth=3,  capthick=3, errorevery=1, alpha=1, ms=4, capsize=5)
            #elif row[1]=="HEPHY":
                #fig2, ax2.bar(row[0], row[2], yerr=row[3], alpha=0.5, ecolor='black', capsize=10, label='Measured at HEPHY', color='green')
            #    fig2, ax2.errorbar(row[0], row[2], yerr=row[3], fmt='o', color='Blue', elinewidth=3, capthick=3, errorevery=1, alpha=1, ms=4, capsize=5)
            #else:
            #    fig2, ax2.bar(row[0], row[2], yerr=row[3], alpha=0.5, ecolor='black', capsize=10, label="Measured at Brown", color='red')
                #fig2, ax2.errorbar(row[0], row[2], yerr=row[3], fmt='o', color='Blue', elinewidth=3, capthick=3, errorevery=1, alpha=1, ms=4, capsize=5)

        # horizontal line indicating the threshold
        #ax2.errorbar(data['Name'], data['Average'], yerr=data['Std.dev'], fmt='o', color='Black', elinewidth=3,
        #             capthick=3,
        #             errorevery=1, alpha=1, ms=4, capsize=5)


        ''''
        for index, row in data.iterrows():
            names=[row[0]]
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
        ax2.set_ylim(0.0,-6.0)
        ax2.set_title(measurment, fontsize=20)
        '''
        '''
            n, bins, patches = ax3.hist(data['Average'], range=(-10,10), bins=50)
            (mu, sigma) = norm.fit(data['Average'])
            # add a 'best fit' line
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            y = norm.pdf(x, mu, sigma)
            l = plt.plot(x, y, 'r--', linewidth=2)

            #plt.legend(handles, labels, prop={'size': 20})
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.xlabel('Batch-number',fontsize=20)
            plt.ylabel('Flatband Voltage [V]', fontsize=20)
            plt.xticks(rotation=45)
        '''



#plt.savefig('.png', bbox_inches='tight')
width = root.winfo_screenwidth()
height = root.winfo_screenheight()
#fig.set_size_inches(width / 100, height / 100)
fig.savefig("Bar_chart" + "_" + measurment + ".png", dpi=100, bbox_inches='tight')

plt.show()

