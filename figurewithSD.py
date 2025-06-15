import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

# Set the backend to pgf
#plt.rcParams["text.usetex"] = True
#plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams.update({'font.size': 13})


# Define the categories and values
categories = ['w/o $C_{1}$', 'w/o $C_{2}$', 'w/o $C_{3}$', 'w/o $C_{4}$']
#means = [84.29, 81.23, 80.78, 82.94, 83.17]
#std_devs = [0.35, 0.51, 0.87, 0.38, 0.70]
means = [79.14, 78.33, 78.11, 79.42]
std_devs = [1.79, 1.99, 1.30, 1.10]

# ICML-style color
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf']#, '#a65628']

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Create the bar chart
bars = ax.bar(range(len(categories)), means, yerr=std_devs, align='center', alpha=0.5, ecolor='black', capsize=10, color=colors)

# Add data labels
for i, bar in enumerate(bars):
    yval = bar.get_height()
    #print('yval = ',yval)
    #print('bar.get_x() = ',bar.get_x())
    #1/0
    ax.text(bar.get_x() + bar.get_width() / 2, 75.5, "{:.2f}".format(yval), ha='center', va='top', fontsize = 18)

# Customise the chart
ax.set_ylabel('Segmentation Results of  (\%)', fontsize = 18)
ax.set_xticks(range(len(categories)))  # add x-axis labels back
ax.set_xticklabels(categories, fontsize = 18)  # set x-axis labels as categories
ax.set_ylim([50, 85])  # Clip the y-axis values
ax.set_yticks([])  # remove y-axis labels

# Save and display the chart
#plt.tight_layout()
plt.savefig('client_contribution.png', dpi = 300)
plt.show()