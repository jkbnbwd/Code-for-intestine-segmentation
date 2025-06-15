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
from networks.vision_transformer import SwinUnet as ViT_seg
from networks.swin3d.SwinUnet_3D import SwinUnet3D
from config import get_config
###model1 trans

parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str,
                    default='/dataT1/Free/qinan/intestine/train/mul-semi', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='test', help='experiment_name')
parser.add_argument('--model2', type=str,
                    default='ri2', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[16, 256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=int,  default=0, help='GPU number')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument(
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=7,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

args = parser.parse_args()
config = get_config(args)
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
            fn = np.sum(gt[i]) - tp
            fp = np.sum(pred[i]) - tp
            precision += tp / (tp+fp+1.0) 
            recall += tp / (tp+fn+1.0)
            #union = np.sum(gt[i]) + np.sum(pred[i])
            dice += (2*tp) / (2*tp+fp+fn+1.0)
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
                #print('model2 y shape = ',y.shape)# torch.Size([1, 2, 16, 256, 256])
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



def test_single_case_2d(trans, image, stride_xy, stride_z, patch_size, num_classes=1):
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
                #print('test_patch shape = ', test_patch.shape) #([1,1,16, 256, 256])
                #print('test_patch[:,:,i] shape = ', test_patch[:,:,0].shape)
                y = torch.zeros([1,2,16,256,256]) #torch.Size([1, 2, 16, 256, 256])
                #print('y shape = ', y.shape)
                with torch.no_grad():
                    for i in range(test_patch.shape[2]):
                        y1 = trans(test_patch[:,:,i])
                        # ensemble
                        y2 = torch.softmax(y1, dim=1)
                        #y2 = y2.unsqueeze(dim=2) #torch.Size([1, 2, 1, 256, 256])
                        #print('y2 shape = ',y2.shape)
                        #print('y2 type = ',type(y2))
                        #print(' y[:,:,i] shape = ',  y[:,:,i].shape)
                        y[:,:,i] = y2
                
                #print('y shape = ', y.shape)                        
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
    patch_size = [16,256,256]
    stride_xy = 8
    stride_z = 256
    save_path = '../val/'+args.exp
    dataname =[]
    #device = torch.device('cpu' if args.cpu else 'cuda')
    # model info
    model1 = ViT_seg(config, img_size=[16,256,156],
                     num_classes=2).cuda()
    model1.load_from(config)
    ckpt1 = torch.load('../model/'+args.exp+'/best_model/best_model1.pth')

    model2 = net_factory_3d(net_type=args.model2, in_chns=1, class_num=2).cuda()
    ckpt2 = torch.load('../model/'+args.exp+'/best_model/best_model2.pth')


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

        pred_np1 = test_single_case_2d(model1, ct_np, stride_xy,stride_z,patch_size,n_labels)
        #print('pred_np1 shape = ', pred_np1.shape)
        
        print('model2 eval...... ')
        pred_np2 = test_single_case(model2, ct_np, stride_xy,stride_z,patch_size,n_labels)


        metric1 = cal_sparse(label_np,pred_np1)
        metric2 = cal_sparse(label_np,pred_np2)

        metric1 =  np.insert(metric1,0,int(data_path[0].split('/')[-1][8:11]))
        metric2 =  np.insert(metric2,0,int(data_path[0].split('/')[-1][8:11]))

        metric1 = np.array(metric1).reshape(1,4)
        metric2 = np.array(metric2).reshape(1,4)
        dataname.append(data_path[0].split('/')[-1])

        col_name = ['ID','dice','recall','precision']
        test1 =  pd.DataFrame(columns = col_name, data=metric1)
        test2 =  pd.DataFrame(columns = col_name, data=metric2)
        pred_img1 = sitk.GetImageFromArray(pred_np1.astype(np.uint8))
        pred_img2 = sitk.GetImageFromArray(pred_np2.astype(np.uint8))

        if not os.path.exists('../result/'+args.exp+'/model1'):
            os.makedirs('../result/'+args.exp+'/model1')
            test1.to_csv('../result/'+args.exp+'/model1/'+'model1_val.csv',encoding='gbk',mode = 'a')
            sitk.WriteImage(pred_img1, os.path.join('../result/'+ args.exp+'/model1', 'test-'+data_path[0].split('/')[-1]))
            
        else:
            test1.to_csv('../result/'+args.exp+'/model1/'+'model1_val.csv',encoding='gbk',mode = 'a',header=False)
            sitk.WriteImage(pred_img1, os.path.join('../result/'+ args.exp+'/model1', 'test-'+data_path[0].split('/')[-1]))

        if not os.path.exists('../result/'+args.exp+'/model2'):
            os.makedirs('../result/'+args.exp+'/model2')
            test2.to_csv('../result/'+args.exp+'/model2/'+'model2_val.csv',encoding='gbk',mode = 'a')
            sitk.WriteImage(pred_img2, os.path.join('../result/'+ args.exp+'/model2', 'test-'+data_path[0].split('/')[-1]))

        else:
            test2.to_csv('../result/'+args.exp+'/model2/'+'model2_val.csv',encoding='gbk',mode = 'a',header=False)
            sitk.WriteImage(pred_img2, os.path.join('../result/'+ args.exp+'/model2', 'test-'+data_path[0].split('/')[-1]))