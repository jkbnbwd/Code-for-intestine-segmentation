import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import itertools
from torch.utils.data.sampler import Sampler
import SimpleITK as sitk


def load_file_name_list(file_path):  
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # 整行读取数据
            if not lines:
                break
            file_name_list.append(lines.split())
    return file_name_list


def randomdelete(file_path):

    txt_path = '/dataT1/Free/qinan/Brats19/train/new_val_path_list.txt'
    f = open(txt_path, 'w')
    file_name_list = load_file_name_list(file_path)
    #print('file_name_list[0] = ', file_name_list[0])
    #1/0
    for i in range(len(file_name_list)):
        label_img = sitk.ReadImage(file_name_list[i][1], sitk.sitkUInt8)
        label = sitk.GetArrayFromImage(label_img)
        if label.sum() == 0 & i%8 == 0:
            continue
        else:
            f.write(file_name_list[i][0]+' '+file_name_list[i][1]+"\n")
            #1/0


if __name__ == "__main__":
    file_path = '/dataT1/Free/qinan/Brats19/train/val_path_list.txt'
    randomdelete(file_path)