#import sys
#import os
from pathlib import Path
#import shutil
from functools import partial

import numpy as np
#import nibabel as nib
import pandas as pd
from tqdm import tqdm
#from p_tqdm import p_map
# package from: https://github.com/deepmind/surface-distance
from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance

import numpy as np
np.bool = np.bool_
import torch
import os
import sys
import pandas as pd
from scipy.ndimage import distance_transform_edt
from medpy import metric
import SimpleITK as sitk
from monai.metrics import SurfaceDistanceMetric
from monai.metrics import SurfaceDiceMetric
#from monai.metrics import compute_meandice
from monai.transforms import AsDiscrete

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def calculate_surface_distances(segmentation, ground_truth, spacing):
    """
    Calculate surface distances between segmentation and ground truth
    """
    surface_distances = []

    segmentation_surface = np.logical_xor(segmentation, distance_transform_edt(segmentation) <= 1)
    ground_truth_surface = np.logical_xor(ground_truth, distance_transform_edt(ground_truth) <= 1)

    segmentation_surface_points = np.argwhere(segmentation_surface)
    ground_truth_surface_points = np.argwhere(ground_truth_surface)

    for point in segmentation_surface_points:
        distances = np.sqrt(np.sum(((ground_truth_surface_points - point) * spacing) ** 2, axis=1))
        surface_distances.append(np.min(distances))

    return np.array(surface_distances)

def calculate_nsd(segmentation, ground_truth, spacing, threshold):
    """
    Calculate the Normalized Surface Dice (NSD) between segmentation and ground truth
    """
    surface_distances = calculate_surface_distances(segmentation, ground_truth, spacing)
    return np.mean(surface_distances < threshold)

# Example usage


if __name__ == '__main__':

    ''' writing CT path to txt file
    dataset_path = os.path.join('./calnsd/MVSNet')
    gt_path = os.path.join('./calnsd/gt')
    #print('dataset_path =',dataset_path)
    #1/0
    fileList=os.listdir(dataset_path)
    #print('myList  = ', fileList)
    gtList=os.listdir(gt_path)
    #1/0
    fileList.sort()
    gtList.sort()

    #nsdtxt = open('./calnsd/6-labeled/pro_nsd_list.txt', 'w')
    nsdtxt = open('./calnsd/mnsnet_nsd_list.txt', 'w')

    for i in range(len(fileList)):

        nsdtxt.write('./calnsd/MVSNet/'+fileList[i]+' '+'/dataT1/Free/qinan/intestine/label/'+gtList[i]+'\n')
    1/0
    '''

    #data_path = os.path.join('./calnsd/pro-best')
    #gt_path = os.path.join('./calnsd/gt')
    #surface_distance = SurfaceDistanceMetric(include_background=False, distance_metric="euclidean", reduction='mean')
    #NSD_cal = SurfaceDiceMetric(class_thresholds=[2.0], include_background=False, distance_metric='euclidean', reduction='mean')
    #dataset_path = ('./calnsd/pro_nsd_list.txt')
    dataset_path = ('./calnsd/total.txt')
    #dataset_path = ('./calnsd/new_nnunet_test_list.txt')
    file_name_list = []
    with open(dataset_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # 整行读取数据
            if not lines:
                break
            file_name_list.append(lines.split())

    for i in range(len(file_name_list)):
        data_path = file_name_list[i]

        ct_path = './calnsd/total/'+file_name_list[i][0]
        label_path = './calnsd/gt/'+file_name_list[i][1]
        #ct_path = file_name_list[i][0]
        #label_path = file_name_list[i][1]

        ct = sitk.ReadImage(ct_path,sitk.sitkInt16)
        ct_np = sitk.GetArrayFromImage(ct)
        
        label = sitk.ReadImage(label_path,sitk.sitkInt8)
        label_np = sitk.GetArrayFromImage(label)

        pred_tensor = torch.tensor(ct_np)
        gt_tensor = torch.tensor(label_np)

        ground_truth = gt_tensor.unsqueeze(0).unsqueeze(0).squeeze(1)
        prediction = pred_tensor.unsqueeze(0).unsqueeze(0).squeeze(1)

        #ground_truth = AsDiscrete(to_onehot=2)(ground_truth).squeeze(1)
        #prediction = AsDiscrete(to_onehot=2)(prediction).squeeze(1)
        #ground_truth = AsDiscrete(to_onehot=2)(ground_truth)
        #prediction = AsDiscrete(to_onehot=2)(prediction)
        #print('ground_truth = ', ground_truth.shape)
        #print('prediction = ', prediction.shape)
        #print('ground_truth = ', ground_truth[:,0,1,1,1])
        #print('prediction = ', prediction[:,0,1,1,1])
        #1/0

        #pred = ct_np
        #gt = label_np
        pred = prediction
        gt = ground_truth
        label_count = 0
        spacing = [1.0, 1.0]
        threshold = 1.0
        nsd = 0.0
        distance_metric = []
        #nsd_value = calculate_nsd(pred, gt, spacing, threshold)
        #ravd = abs(metric.binary.ravd(pred, gt))
        #nsd = surface_distance(y_pred=prediction, y=ground_truth, distance_threshold=1.0)
        #nsd = NSD_cal(y_pred=prediction, y=ground_truth)
        #sd = compute_surface_distances(np.array(gt[0],dtype=bool),np.array(pred[0],dtype=bool), [1.0,1.0,1.0])
        #nsd = compute_surface_dice_at_tolerance(sd, 1.0)
        
        #'''
        nsd_sum = 0.0
        for j in range(0,gt.shape[2]):#循环每张slice，并判断是否被标注
            #print('j == ',j, 'nsd original = ', nsd)
            #print('gt.shape = ', gt.shape[2]) 
            #1/0  
            if gt[:,:,j].max() != gt[:,:,j].min():
                label_count += 1
                if pred[:,:,j].max() == 0:
                    continue
                else:
                    #print('nsd befor add new slice = ', nsd)
                    #nsd = surface_distance(y_pred=prediction, y=ground_truth, distance_threshold=1.0)
                    sd = compute_surface_distances(np.array(gt[0,:,j],dtype=bool), np.array(pred[0,:,j],dtype=bool), [1.0, 1.0])
                    nsd = compute_surface_dice_at_tolerance(sd, 3.0)
                    #print('nsd cal = ',nsd)
                    #1/0
                    
                    #print('nsd after add new slice = ', nsd_sum)
                    
                    #nsd = NSD_cal(y_pred=pred[:,:,j], y=gt[:,:,j])
                    nsd_sum += nsd
                    #print('sd shape = ',sd.shape)
                    #print('nsd = ',nsd)

         
        #print('nsd sum = ', nsd_sum)
        #1/0
        #print('finish one...')
        if label_count == 0:
            nsd_sum = nsd_sum / 1
        else:
            #print('nsd111 = ',nsd)
            nsd_sum = nsd_sum / label_count
            #print('label_count = ',label_count)
            #print('nsd final = ',nsd_sum)
            #1/0
        #'''

        #print('label_count = ',label_count)
        #print('nsd = ', nsd)
        #print('torch.mean(nsd)= ', torch.mean(nsd))
        #1/0 
        distance_metric.append(i)
        #print(' distance_metric type = ',type(distance_metric))
        #distance_metric.append(torch.mean(nsd))
        #print('nsd.numpy()[0] = ', nsd.numpy()[0])
        #print('nsd.numpy()[0] = ', type(nsd.numpy()[0]))
        #print('nsd.numpy()[0] = ', type(torch.mean(nsd)))
        #distance_metric.append(torch.from_numpy(nsd.numpy()[0]))
        #distance_metric.append(float(nsd_sum.numpy()[0]))
        distance_metric.append(nsd_sum)
        #print('np.array(distance_metric).shape ',np.array(distance_metric).shape)
        #1/0
        distance_metric = np.array(distance_metric).reshape(1,2)
        col_name = ['ID','nsd']
        dis_metric = pd.DataFrame(columns = col_name, data=distance_metric)
        dis_metric.to_csv('./calnsd/13-labeled/newnsd/total_nsd_list_1202.csv',encoding='gbk',mode = 'a',header=False)
        print('finish '+ file_name_list[i][1]+' ...')
        #1/0


