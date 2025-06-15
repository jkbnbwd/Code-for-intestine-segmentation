import torch
import scipy.stats as stats
from scipy.stats import wilcoxon

import pandas as pd

# Assuming you have two tensors with your data
p_list1 = []
w_list = []
p_list = []
csv_file = './pvalue-rebuttal.csv'
data_frame = pd.read_csv(csv_file)
data_tensor = torch.tensor(data_frame.values, dtype=torch.float32)
#print('data_tensor shape = ',data_tensor.shape)
#1/0
#data_tensor1 = data_tensor[:,1:] # to remove the first colume (all value =0) and get the rest data to a new tensor.
#print('data_tensor1 shape = ',data_tensor1.shape)
#1/0
data_tensor1 = data_tensor
data_sample2 = data_tensor1[:,1]
#print('data_sample2= ', data_sample2)
#1/0
data_sample2_np = data_sample2.numpy()
for i in range(2):
    data_sample1 = data_tensor1[:,i]
    #data_sample2 = data_tensor1[:,1]
    data_sample1_np = data_sample1.numpy()
    
    t_stat, p_value = stats.ttest_ind(data_sample1_np, data_sample2_np)
    p_list.append(p_value)
    t_stat, p_value1 = stats.ttest_rel(data_sample1_np, data_sample2_np)
    p_list1.append(p_value1)
  



print("P-list:", p_list)
print("P-list1:", p_list1)