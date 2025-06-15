#from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import config
#from utils import logger,common
import SimpleITK as sitk
import os
import numpy as np
from networks.net_factory_3d import net_factory_3d
#from collections import OrderedDict
#from medpy import metric
from PIL import Image
import math
import SimpleITK as sitk
import time
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--csvname', type=str,
                    default='data', help='experiment_name')
parser.add_argument('--model1', type=str,
                    default='ri2', help='model_name') #unet_3D
parser.add_argument('--model2', type=str,
                    default='ri2', help='model_name')
parser.add_argument('--upper', type=int,
                    default=200, help='ct value')
parser.add_argument('--lower', type=int,
                    default=-100, help='ct value')
parser.add_argument('--gpu', type=int,
                    default=0, help='ct value')
args = parser.parse_args()

gpu = str(args.gpu)

os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def to_one_hot_3d(tensor, n_classes=2):  # shape = [batch, s, h, w]
    n, s, h, w = tensor.size()
    one_hot = torch.zeros(n, n_classes, s, h, w).scatter_(1, tensor.view(n, 1, s, h, w), 1)
    return one_hot

def cal_metric(gt, pred):
    metric = []

    tp = np.sum(gt * pred)
    fp = np.sum(gt) - tp 
    fn = np.sum(pred) - tp
    dice = (2*tp) / (2*tp+fp+fn+1.0) 
    recall = tp / (tp+fp+1.0)  #sensitiviry
    precision = tp / (tp+fn+1.0) 
    #metric.append(iter_num)
    metric.append(dice)
    metric.append(recall)
    metric.append(precision)
   
    return metric

def cal_sparse(gt, pred):
    #print('pred shape = ', pred.shape)
    #print('gt shape = ', gt.shape)
    #pred = pred.squeeze(0)

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
            precision += tp / (tp+fp+1.0) 
            recall += tp / (tp+fn+1.0)
            #union = np.sum(gt[i]) + np.sum(pred[i])
            dice += (2*tp) / (2*tp+fp+fn+1.0)
            #dice2 += 2*recall*precision / (recall + precision+1.0)
            #print('dice1 = ',dice1)
            #print('dice2 = ',dice2)
            total_tp += tp
            total_fp += fp
            total_fn += fn


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


def connect(img,img_np,mask = True): #connection area
    img_nd = sitk.GetArrayFromImage(img)
    #print('img_nd shape = ',img_nd.shape)
    #print('imgh shape = ',img.shape)
    #1/0
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    img_out = cca.Execute(img)
    img_filter = sitk.LabelShapeStatisticsImageFilter()
    img_filter.Execute(img_out)
    num_connect_label = cca.GetObjectCount()
    print('num_label = ',num_connect_label)
    area_max = 300
    area_list = []
    area_max_label = 0
    con_mask = np.zeros(img_np.shape)
    for i in range(1,num_connect_label+1):
        print('i =',i)
        area = img_filter.GetNumberOfPixels(i)
        print('area = ',area)
        if area > area_max:
            star =  time.time()
            mask = np.where(img == i,1,0)
            con_mask = np.maximum(mask,con_mask)
            end = time.time()
            print('con time = ',(end -  star)//60)
    print('for 1 end !!!')  
    print('con_mask shape ',con_mask.shape)
    print('con_mask type',type(con_mask))
    con_img = sitk.GetImageFromArray(con_mask)
    con_img.SetOrigin(img.GetOrigin())
    con_img.SetSpacing(img.GetSpacing())
    con_img.SetDirection(img.GetDirection())
    sitk.WriteImage(con_img, 'result-'+data_path[0].split('/')[-1])
    return con_img
# 1/0

def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    #print('image.shape  = ', image.shape )
    #1/0

    w, h, d = image.shape 

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
    sy = math.ceil((hh - patch_size[1]) / stride_z) + 1
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
    #print('cnt  shape = ', cnt.shape) 
    #score_map = score_map/np.expand_dims(cnt, axis=0)
    score_map = score_map/cnt   #.unsqueeze(dim=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    
    return label_map


if __name__ == '__main__':
    
    n_labels = 2
    patch_size = [16,256,256] #test_cut_size
    #patch_stride = [8,128,128] #test_cut_stride
    stride_xy = 8
    stride_z = 256
    save_path = '../val/'+args.csvname
    dataname =[]
    #device = torch.device('cpu' if args.cpu else 'cuda')
    # model info
    model1 = net_factory_3d(net_type=args.model1, in_chns=1, class_num=2).cuda()
    ckpt1 = torch.load('../model/'+args.csvname+'/best_model/best_model1.pth')
    #ckpt1 = torch.load('../model/inte-0322-01/unet_3D/model1_iter_3000.pth')
    model2 = net_factory_3d(net_type=args.model2, in_chns=1, class_num=2).cuda()
    ckpt2 = torch.load('../model/'+args.csvname+'/best_model/best_model2.pth')
    #ckpt2 = torch.load('../model/inte-0322-01/unet_3D/model1_iter_3000.pth')

    model1.load_state_dict(ckpt1)
    model1.eval()
    model2.load_state_dict(ckpt2)
    model2.eval()

    # save data info
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    dataset_path = os.path.join('/dataT1/Free/qinan/intestine/test_path_list.txt')
    #print('dataset_path =',dataset_path)
    #1/0
    file_name_list = []
    with open(dataset_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # 整行读取数据
            if not lines:
                break
            file_name_list.append(lines.split())
    for i in range(len(file_name_list)):
        data_path = file_name_list[i]

        ct_path = file_name_list[i][0]
        label_path = file_name_list[i][1]

        ct = sitk.ReadImage(ct_path,sitk.sitkInt16)
        ct_np = sitk.GetArrayFromImage(ct)
        
        label = sitk.ReadImage(label_path,sitk.sitkInt8)
        label_np = sitk.GetArrayFromImage(label)
        label_np[label_np>0] = 1
        #print('ct_np shape = ', ct_np.shape)
        #print('label_np shape = ', label_np.shape)
        print("Start Evaluate: ", data_path[0].split('/')[-1],str(i+1)+'/'+str(len(file_name_list)))

        pred_np1 = test_single_case(model1, ct_np, stride_xy,stride_z,patch_size,n_labels)
        pred_np2 = test_single_case(model2, ct_np, stride_xy,stride_z,patch_size,n_labels)

        #metric1 = cal_metric(label_np,pred_np1)
        #metric2 = cal_metric(label_np,pred_np2)
        metric1 = cal_sparse(label_np,pred_np1)
        metric2 = cal_sparse(label_np,pred_np2)
        #print('metric  = ',metric)
        metric1 =  np.insert(metric1,0,int(data_path[0].split('/')[-1][8:11]))
        metric2 =  np.insert(metric2,0,int(data_path[0].split('/')[-1][8:11]))
        #print('data_path[0].split(/)[-1][8:10] = ', data_path[0].split('/')[-1][8:11])
        #1/0
        metric1 = np.array(metric1).reshape(1,4)
        metric2 = np.array(metric2).reshape(1,4)
        dataname.append(data_path[0].split('/')[-1])

        col_name = ['ID','dice','recall','precision']
        test1 =  pd.DataFrame(columns = col_name, data=metric1)
        test2 =  pd.DataFrame(columns = col_name, data=metric2)
        #pred_img1 = sitk.GetImageFromArray(np.squeeze(pred_np1,axis=0).astype(np.uint8))
        #pred_img2 = sitk.GetImageFromArray(np.squeeze(pred_np2,axis=0).astype(np.uint8))
        pred_img1 = sitk.GetImageFromArray(pred_np1.astype(np.uint8))
        pred_img2 = sitk.GetImageFromArray(pred_np2.astype(np.uint8))

        if not os.path.exists('../result/'+args.csvname+'/model1'):
            os.makedirs('../result/'+args.csvname+'/model1')
            test1.to_csv('../result/'+args.csvname+'/model1/'+'model1_val.csv',encoding='gbk',mode = 'a')
            sitk.WriteImage(pred_img1, os.path.join('../result/'+ args.csvname+'/model1', 'test-'+data_path[0].split('/')[-1]))
            #sitk.WriteImage(pred_img1, './test.nii.gz')
        else:
            test1.to_csv('../result/'+args.csvname+'/model1/'+'model1_val.csv',encoding='gbk',mode = 'a',header=False)
            sitk.WriteImage(pred_img1, os.path.join('../result/'+ args.csvname+'/model1', 'test-'+data_path[0].split('/')[-1]))
            #sitk.WriteImage(pred_img1, './test.nii.gz')

        if not os.path.exists('../result/'+args.csvname+'/model2'):
            os.makedirs('../result/'+args.csvname+'/model2')
            test2.to_csv('../result/'+args.csvname+'/model2/'+'model2_val.csv',encoding='gbk',mode = 'a')
            sitk.WriteImage(pred_img2, os.path.join('../result/'+ args.csvname+'/model2', 'test-'+data_path[0].split('/')[-1]))
            #sitk.WriteImage(pred_img1, './test.nii.gz')
        else:
            test2.to_csv('../result/'+args.csvname+'/model2/'+'model2_val.csv',encoding='gbk',mode = 'a',header=False)
            sitk.WriteImage(pred_img2, os.path.join('../result/'+ args.csvname+'/model2', 'test-'+data_path[0].split('/')[-1]))
            #sitk.WriteImage(pred_img1, './test.nii.gz')
        #pred_img = sitk.GetImageFromArray(np.squeeze(pred_np,axis=0).astype(np.uint8))
    
        #if not os.path.exists('../result/'+ args.csvname):
        #    os.makedirs('../result/'+ args.csvname)
        #sitk.WriteImage(pred_img, os.path.join('../result/'+ args.csvname, 'test-model2-'+data_path[0].split('/')[-1]))
       