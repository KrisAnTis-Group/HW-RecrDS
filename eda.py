import numpy as np
import matplotlib.pyplot as plt
from random import randint

import src 
from src import edaPlotDataAdapt as DA

def scatterPlot(x_data, y_data, x_label="", y_label="", title="", color = "r", yscale_log=False):

    # Create the plot object
    _, ax = plt.subplots()

    # Plot the data, set the size (s), color and transparency (alpha)
    # of the points
    ax.scatter(x_data, y_data, s = 10, color = color, alpha = 0.75)

    if yscale_log == True:
        ax.set_yscale('log')

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)



def histogram(data, n_bins, cumulative=False, x_label = "", y_label = "", title = ""):
    _, bx = plt.subplots()
    bx.hist(data, bins = n_bins, cumulative = cumulative, color = '#539caf')
    bx.set_ylabel(y_label)
    bx.set_xlabel(x_label)
    bx.set_title(title)

def linePlot( x_data, y_data, x_label="", y_label="", title=""):
    # Create the plot object
    _, ax = plt.subplots()

    # Plot the best fit line, set the linewidth (lw), color and
    # transparency (alpha) of the line
    ax.plot( x_data, y_data, lw = 1, color = '#539caf', alpha = 1)

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def barPlot(x_data, y_data, x_label="", y_label="", title=""):
    _, ax = plt.subplots()
    # Draw bars, position them in the center of the tick mark on the x-axis
    ax.bar(x_data, y_data, color = '#539caf', align = 'center')
    # Draw error bars to show standard deviation, set ls to 'none'
    # to remove line between points
   # ax.errorbar(x_data,y_data, color = '#297083', ls = 'none', lw = 2, capthick = 2)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)

def groupedBarPlot(y_data_list, y_data_names="", x_label="", y_label="", title=""):
    _, ax = plt.subplots()
    # Total width for all bars at one x location
    total_width = 0.8
    # Width of each individual bar
    ind_width = total_width / len(y_data_list)
    spase_width = (1-total_width)/ len(y_data_list)
    # This centers each cluster of bars about the x tick mark
    alteration = np.arange(-1, 1, ind_width+spase_width)

    # Draw bars, one category at a time
    for q in range(0, len(y_data_list)):
        colors = DA.getColors(len(y_data_list[q]))
        for i in range(1, len(y_data_list[q])):
            ax.bar(alteration[q], np.sum(y_data_list[q][:i]),  bottom =  np.sum(y_data_list[q][:i-1]),  color = colors[i], width = ind_width)
            
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc = 'upper right')

def groupedBar (data, plots = None):
    if not plots:
        plots = range(len(data))
    
    groupedBarPlot(data[plots.start:plots.stop])




