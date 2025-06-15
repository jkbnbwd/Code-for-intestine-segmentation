

'''
##SSL BOXPLOT
import torch
import matplotlib.pyplot as plt
#import seaborn as sns
 
# import the numpy package
import numpy as np
import pandas as pd
import matplotlib
 
# create 2 - sample a 3-Dim array, that measures
# the summer and winter rain fall amount
csv_file1 = './pvalue-boxplot-13.csv'
data_frame1 = pd.read_csv(csv_file1)
csv_file2 = './pvalue-boxplot-06.csv'
data_frame2 = pd.read_csv(csv_file2)
data_tensor_13 = torch.tensor(data_frame1.values, dtype=torch.float32)
data_tensor_06 = torch.tensor(data_frame2.values, dtype=torch.float32)
data_tensor_13 = data_tensor_13 * 100
data_tensor_06 = data_tensor_06 * 100
train_6 = data_tensor_06.reshape(7,-1).tolist()
train_13 = data_tensor_13.reshape(7,-1).tolist()
#train_6 = [train_6*100 for i in ra]
#train_13 = train_13*100
#print('train_6 = ', train_6)
#1/0
matplotlib.rcParams['font.family'] = 'Times New Roman'

 
# the list named ticks, summarizes or groups
# the summer and winter rainfall as low, mid
# and high
ticks = ['3D U-Net', 'CPS', 'EM', 'MT', 'Qin','Swin','Proposed']
 
# create a boxplot for two arrays separately,
# the position specifies the location of the
# particular box in the graph,
# this can be changed as per your wish. Use width
# to specify the width of the plot
six_plot = plt.violinplot(train_6,positions=np.array(np.arange(len(train_6)))*2.0-0.35,
                   widths=0.6, showmeans=False,showmedians=True)
thir_plot = plt.violinplot(train_13,positions=np.array(np.arange(len(train_13)))*2.0+0.35,
                   widths=0.6, showmeans=False,showmedians=True)
 
# each plot returns a dictionary, use plt.setp()
# function to assign the color code
# for all properties of the box plot of particular group
# use the below function to set color for particular group,
# by iterating over all properties of the box plot
def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)


    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code, label=label)
    if flag:
        #plt.plot([],marker='*',c='k', label = 'P-value < 0.05')
        plt.scatter(2.0+1,93+1.7,marker='*',c='k',label = 'P-value < 0.05')
    plt.legend(loc = 2, bbox_to_anchor=(1.05,1), borderaxespad = 0)

    x = np.arange((len(train_13))*2.0+0.35)
    for i in range(len(train_13)):
        plt.plot([i*2.0+0.35,6*2.0+0.35],[93+i*1,93+i*1], linewidth = 1, c='k')
        if i < 6:
            plt.scatter(i*2.0+1,93+i*1+1,marker='*', c='k')
        
        if i==6:
            plt.plot([i*2.0+0.35,i*2.0+0.35],[92,93+i*1-1], linewidth = 1, c='k')
        else:
            plt.plot([i*2.0+0.35,i*2.0+0.35],[92,93+i*1], linewidth = 1, c='k')
        
 
# setting colors for each groups
flag = 0
define_box_properties(six_plot, '#D7191C', '6 labeled cases')
flag = 1
define_box_properties(thir_plot, '#2C7BB6', '13 labeled cases')

# set the x label values
plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks)
plt.yticks(range(0, 100,5))
plt.xlabel('Methods')
plt.ylabel('Dice (%)')
# set the limit for x axis
plt.xlim(-2, len(ticks)*2)
#y_values = [x*5 for x in range(18)]

 
# set the limit for y axis
plt.ylim(0, 100)
 
# set the title
#plt.title('Grouped boxplot Dice scores for different methods')
plt.savefig('gboxplot-ssl.jpg',dpi=600,bbox_inches='tight')
plt.savefig('gboxplot-ssl.pdf',dpi=600,bbox_inches='tight')
'''
'''
import matplotlib

matplotlib.rcParams['font.family'] = 'Times New Roman'



import matplotlib.pyplot as plt
 
# import the numpy package
import numpy as np
 
# create 2 - sample a 3-Dim array, that measures
# the summer and winter rain fall amount
train_6 = [[52.65,45.26,35.97,43.92,53.22],[71.70,73.12,74.46,73.12,72.56],\
            [69.26,68.97,70.03,72.34,66.65],[71.79,68.30,74.71,59.50,74.33],\
                [78.65,73.67,78.41,78.36,76.83]]
train_13 = [[71.23,39.66,43.80,36.43,42.87],[77.34,77.95,78.44,69.20,77.05],\
            [78.39,72.24,78.26,77.69,73.89],[75.80,77.38,77.28,61.92,77.87],\
                [81.54,79.86,80.18,79.90,80.76]]
 
# the list named ticks, summarizes or groups
# the summer and winter rainfall as low, mid
# and high
#print('type train_6 = ',type(train_6))
#1/0
ticks = ['3D U-Net', 'CPS', 'EM', 'MT', 'Proposed']
 
# create a boxplot for two arrays separately,
# the position specifies the location of the
# particular box in the graph,
# this can be changed as per your wish. Use width
# to specify the width of the plot
six_plot = plt.boxplot(train_6,
                               positions=np.array(
    np.arange(len(train_6)))*2.0-0.35,
                               widths=0.6)
thir_plot = plt.boxplot(train_13,
                               positions=np.array(
    np.arange(len(train_13)))*2.0+0.35,
                               widths=0.6)
 
# each plot returns a dictionary, use plt.setp()
# function to assign the color code
# for all properties of the box plot of particular group
# use the below function to set color for particular group,
# by iterating over all properties of the box plot
def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)
    
         
    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code, label=label)
    if flag:
        #plt.plot([],marker='*',c='k', label = 'P-value < 0.05')
        plt.scatter(3*2.0+0,33,marker='*',c='k',label = 'P-value < 0.05')
    plt.legend(loc = 4)

    x = np.arange((len(train_13))*2.0+0.35)
    for i in range(len(train_13)):
        plt.plot([i*2.0+0.35,4*2.0+0.35],[84+i*1,84+i*1], linewidth = 1, c='k')
        if i < 4:
            plt.scatter(i*2.0+1,84+i*1+1,marker='*', c='k')
        
        if i==4:
            plt.plot([i*2.0+0.35,i*2.0+0.35],[82,84+i*1-1], linewidth = 1, c='k')
        else:
            plt.plot([i*2.0+0.35,i*2.0+0.35],[82,84+i*1], linewidth = 1, c='k')
        
 
# setting colors for each groups
flag = 0
define_box_properties(six_plot, '#D7191C', '6 labeled cases')
flag = 1
define_box_properties(thir_plot, '#2C7BB6', '13 labeled cases')


# set the x label values
plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks)
plt.yticks(range(30, 95, 5))
plt.xlabel('Methods')
plt.ylabel('Dice (%)')
# set the limit for x axis
plt.xlim(-2, len(ticks)*2)
y_values = [x*5 for x in range(18)]

 
# set the limit for y axis
plt.ylim(30, 90)
 
# set the title
#plt.title('Grouped boxplot Dice scores for different methods')
plt.savefig('gboxplot-new.jpg',dpi=600,bbox_inches='tight')
plt.savefig('gboxplot-new.pdf',dpi=600,bbox_inches='tight')
'''



import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams['font.family'] = 'Times New Roman'

x_labels = ['3 Cases', '6 Cases', '9 Cases', '13 Cases']
#train_3 = [[64.97, 77.78], [61.58, 80.93], [73.09, 77.01]]
#train_6 = [[72.34, 79.83], [84.27, 82.82], [66.65, 79.04]]
#train_9 = [[69.74, 81.71], [66.62, 83.35], [76.28, 81.71]]
#train_13 = [[80.57, 81.75], [83.19, 83.19], [79.95, 81.90]]
train_3 = [64.97, 77.78]
train_6 = [72.34, 79.83]
train_9 = [69.74, 81.71]
train_13 = [80.57, 81.75]
swin = [64.97,72.34,69.74,80.57]
pro = [77.78, 79.83, 81.71, 81.75]

# Convert data to numpy arrays for easier manipulation
train_3 = np.array(train_3)
train_6 = np.array(train_6)
train_9 = np.array(train_9)
train_13 = np.array(train_13)

# Extract data for each method
method1_3 = train_3[ 0]
method1_6 = train_6[0]
method1_9 = train_9[ 0]
method1_13 = train_13[0]

method2_3 = train_3[ 1]
method2_6 = train_6[1]
method2_9 = train_9[1]
method2_13 = train_13[ 1]
swin = [9.63,9.88,8.64,8.20]
pro = [9.72, 8.86,7.65,7.65]

# Plotting
fig, ax = plt.subplots()

# Define bar width
bar_width = 0.35
index = np.arange(len(x_labels))
train_13 = [method1_3, method1_6, method1_9, method1_13]

# Plot bars for Method 1 (2D Swin U-Net)
x_values = [method1_3, method1_6, method1_9, method1_13]
rects1 = ax.bar(index - bar_width/2, [method1_3, method1_6, method1_9, method1_13],
                bar_width, yerr = swin,capsize=4, label='2D Swin U-Net',)

# Plot bars for Method 2 (Proposed Method)
rects2 = ax.bar(index + bar_width/2, [method2_3, method2_6, method2_9, method2_13], 
                bar_width, yerr = pro,capsize=4, label='Proposed Method',)
ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

for i in range(len(train_13)):
        plt.plot([i*2.0+0.35,6*2.0+0.35],[93+i*1,93+i*1], linewidth = 1, c='k')
        if i < 6:
            plt.scatter(i*2.0+1,93+i*1+1,marker='*', c='k')
        
        if i==6:
            plt.plot([i*2.0+0.35,i*2.0+0.35],[92,93+i*1-1], linewidth = 1, c='k')
        else:
            plt.plot([i*2.0+0.35,i*2.0+0.35],[92,93+i*1], linewidth = 1, c='k')

# Add labels, title, and legend
ax.set_xlabel('Number of Labeled Cases in Training Dataset')
ax.set_ylabel('Dice Score (%)')
#ax.set_title('Dice score of 2D Swin U-Net and the proposed method using different number of labeled cases')
ax.set_xticks(index)
ax.set_xticklabels(x_labels)
ax.legend(loc=4)


#plt.title('Histogram of Training Data')
plt.savefig('response-sd.jpg',dpi=600,bbox_inches='tight')
plt.savefig('response-sd.pdf',dpi=600,bbox_inches='tight')
plt.xlabel('Value')