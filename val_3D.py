import math
from glob import glob

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from tqdm import tqdm
import os
import pandas as pd
import csv

def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):

    w, h, d = image.shape 
    #print('image.shape = ',image.shape) #[16,256,256]
    #print('patch_size = ',patch_size)  #[256,256,16]
    #print('stride_xy  = ',stride_xy) #128
    #print('stride_xy  = ',stride_z) #8
    #1/0
    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = net(test_patch)
                    # ensemble
                    y = torch.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    
    
    return label_map

#'''## for sparse label
def cal_sparse(gt, pred):

    label_count = 0
    dice = 0.0
    recall = 0.0
    precision = 0.0
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    metric = []

    for i in range(0,gt.shape[0]):#循环每张slice，并判断是否被标注   
        
        if gt[i].max() != gt[i].min():
            #print('labeled slice !!!!!')
            label_count += 1
           
            tp = np.sum(gt[i] * pred[i])
            fp = np.sum(gt[i]) - tp
            fn = np.sum(pred[i]) - tp
            tn = gt.shape[1]*gt.shape[2] - tp - fp - fn
            recall += tp / (tp+fp+1.0) #sensitiviry
            precision += tp / (tp+fn+1.0)
            #union = np.sum(gt[i]) + np.sum(pred[i])
            dice += (2*tp) / (2*tp+fp+fn+1.0)
            #dice2 += 2*recall*precision / (recall + precision+1.0)
            #print('dice1 = ',dice1)
            #print('dice2 = ',dice2)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn


    if label_count == 0:
        dice = dice / 1
        recall = recall / 1
        precision = precision / 1
    else:
        dice = dice / label_count
        recall = recall / label_count
        precision = precision / label_count
    #metric.append(id)
    metric.append(dice)
    metric.append(recall)
    metric.append(precision)
    #metric.append(total_tp)
    #metric.append(total_fp)
    #metric.append(total_tn)
    #metric.append(total_fn)
    return metric
#'''


##version2
def cal_metric(gt, pred):
    metric = []

    tp = np.sum(gt * pred)
    fp = np.sum(gt) - tp
    fn = np.sum(pred) - tp
    dice = (2*tp) / (2*tp+fp+fn+1.0)
    recall = tp / (tp+fp+1.0) #sensitiviry
    precision = tp / (tp+fn+1.0)
    #metric.append(iter_num)
    metric.append(dice)
    metric.append(recall)
    metric.append(precision)
   
    return metric




'''
def test_all_case(net, base_dir, test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160), stride_xy=32, stride_z=24):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    image_list = [base_dir + "/data/{}.h5".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((num_classes-1, 2))
    print("Validation begin")
    for image_path in tqdm(image_list):
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction = test_single_case(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        for i in range(1, num_classes):
            total_metric[i-1, :] += cal_metric(label == i, prediction == i)
'''
'''
def writecsv(csv_path, id, data):
    with open (csv_path, mode = 'w') as csv_file:
        writer = csv.writer(csv_file,deliniter=',',quotechar='"', quoting = csv.QUOTE_MINIMAL)
        writer.writerow('ID','Dice','Recall','Precision')
        for i in range(len(data)):
            writer.writerow([id,data])
        #writer.to_csv('./result.csv',index = False)
'''



def test_all_case(csvname, net, base_dir, test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160), \
                  stride_xy=32, stride_z=24, tar = 'default'):
    
    best_dice = 0.0
    cur_dice = 0.0
    image_list = []
    with open(os.path.join(base_dir,test_list), 'r') as f:
        while True:
            lines = f.readline().strip()  # 整行读取数据
            if not lines:
                break
            image_list.append(lines.split())

    total_metric = np.zeros((num_classes-1,3))
    #metric = []
    #print(' total_metric shape = ', total_metric.shape) #(1,2) total_metric = ([0,0])
    print("Validation begin")

    for image_path in tqdm(image_list):
        #print('image_path = ',image_path)
        #print('image_path[0] = ',image_path[0])
        #print('image_path[1] = ',image_path[1])
        #1/0
        #print('image_name  = ', image_path[1].split('/')[-1])
        #1/0
        img = sitk.ReadImage(image_path[0], sitk.sitkFloat32)
        lab = sitk.ReadImage(image_path[1], sitk.sitkUInt8)
        img = sitk.GetArrayFromImage(img)
        lab = sitk.GetArrayFromImage(lab)
        #print('label shape = ',lab.shape)
        #print('img shape = ',img.shape)
        #1/0
        crop_x = int(img.shape[0] / 2)
        crop_y = int(img.shape[1] / 2)
        crop_z = int(img.shape[2] / 2)
        #print()

        image = img[crop_x-8:crop_x+8, crop_y-125:crop_y+125, crop_z-125:crop_z+125]
        label = lab[crop_x-8:crop_x+8, crop_y-125:crop_y+125, crop_z-125:crop_z+125]
    
        prediction = test_single_case(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        pred_image = sitk.GetImageFromArray(prediction.astype(np.uint8))
        #print('image_path[1].split(''/'')[-1] type = ', type(image_path[1].split('/')[-1]))
        #print('222csvname type = ', type(csvname))
        
        #print('prediction shape = ',prediction.shape) #(355, 372, 372)
        
        #id = image_path[1].split('/')[-1]
        #iter_num = iter_num
        metric = cal_metric(label,prediction)
        #metric = cal_sparse(label,prediction)
        #print('metric = ', metric)
        #1/0
        

        #total_dice += dice
        metric = np.array(metric).reshape(1,3)
        #print('metric type = ',type(metric))
        #print('metric = ',metric)
        #1/0
        
        total_metric= np.concatenate((total_metric,metric),axis=0)
        if cur_dice >= best_dice:
            cur_dice = best_dice
            if not os.path.exists('../val/'+ csvname +'/'+tar+'/'):
                    os.makedirs('../val/'+ csvname +'/'+tar+'/')
            sitk.WriteImage(pred_image, '../val/'+ csvname +'/'+tar+'/'+image_path[1].split('/')[-1])
        
        
        #print('total_metric len = ',len(total_metric))
        #print('total_metric type = ', type(total_metric))
        #print('total_metric = ', total_metric)
        #print('dice = ', dice)
        #1/0
        #for i in range(1, num_classes):
        #    total_metric[i-1, :] += cal_metric(label == i, prediction == i)
    #col_name = ['iter_num','dice','recall','precision']
    total_metric = total_metric[1:]
    avg_metric = np.around(np.mean(total_metric,axis=0),4) #averaged of dice, recall and precision
    cur_dice = avg_metric[0]
    #print('avg_metric = ', avg_metric)
    #test =  pd.DataFrame(columns = col_name, data=total_metric)
    #test.to_csv('./test4.csv',encoding='gbk',mode = 'a')
    #clearprint('test = ',test)
    #print('dice / len(image_list) = ', dice / len(image_list))
    
    print("Validation end")
    return avg_metric
    #return total_dice / len(image_list), total_metric
