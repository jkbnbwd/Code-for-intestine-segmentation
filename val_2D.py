import numpy as np
import torch
import SimpleITK as sitk
from medpy import metric
from scipy.ndimage import zoom
import torch.nn.functional as F
from medpy import metric
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str,
                    default='/dataT1/Free/qinan/intestine/train/mul-semi', help='Name of Experiment')
parser.add_argument('--exp', type=str,default='test', help='experiment_name')
parser.add_argument('--gpu', type=int,  default=0, help='GPU number')
args = parser.parse_args()

gpu = str(args.gpu)

os.environ["CUDA_VISIBLE_DEVICES"] = gpu

def cal_metric(pred, gt):  
    metric = []
    tp = torch.sum(gt * pred)
    fp = torch.sum(gt) - tp
    fn = torch.sum(pred) - tp
    dice = (2*tp) / (2*tp+fp+fn+1.0)
    recall = tp / (tp+fp+1.0) #sensitiviry
    precision = tp / (tp+fn+1.0)
    metric.append(dice)
    metric.append(recall)
    metric.append(precision)


    return metric
    #else:
    #    return 0, 0

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
           
            tp = torch.sum(gt[i] * pred[i])
            fp = torch.sum(gt[i]) - tp
            fn = torch.sum(pred[i]) - tp
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
    return metric


def test_single_volume_swin(image, label, net, num_classes, patch_size=[256, 256]):

    prediction = torch.zeros_like(label) #torch.Size([1, 16, 256, 256])

    for i in range(image.shape[0]):

        input = image[i].unsqueeze(0).unsqueeze(0).float().cuda()
        #input = torch.from_numpy(slice).float().cuda()
        #print('input shape = ', input.shape)
        #1/0
        net.eval()
        #media_fea, output = net(input)
        output = net(input)
        with torch.no_grad():
            out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            #out = out.cpu().detach().numpy() #256*256
            #print('out shape = ', out.shape)
            #1/0
            #pred = zoom(out, (x / patch_size[0],  y / patch_size[1]), order=0)
            prediction[i] = out
            #prediction = pred
    #print('prediction shape 111 = ', prediction.shape)
    #print("prediction shape = ", prediction.shape)
    #print('label shape = ', label.shape)
    #1/0
    metric = cal_sparse(label,prediction)
    #print('metric = ', metric)
    return metric



def test_single_volume(image, label, net, num_classes, patch_size=[256, 256]):

    prediction = torch.zeros_like(label) #torch.Size([1, 16, 256, 256])

    for i in range(image.shape[0]):

        input = image[i].unsqueeze(0).unsqueeze(0).float().cuda()
        #input = torch.from_numpy(slice).float().cuda()
        #print('input shape = ', input.shape)
        #1/0
        net.eval()
        #media, output = net(input)
        output = net(input)
        with torch.no_grad():
            out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            prediction[i] = out
            #prediction = pred
    #print('prediction shape 111 = ', prediction.shape)
    return prediction


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(cal_metric(
            prediction == i, label == i))
    return metric_list

def test_all_case_2d(csvname, net, base_dir, num_classes=4, patch_size=(256,256), \
                  stride_xy=32, stride_z=24, tar = 'default'):
    
    best_dice = 0.0
    cur_dice = 0.0
    image_list = []
    #with open(os.path.join(base_dir,test_list), 'r') as f:
    with open(base_dir, 'r') as f:
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

        image = torch.from_numpy(img[crop_x-8:crop_x+8, crop_y-128:crop_y+128, crop_z-128:crop_z+128])
        label = torch.from_numpy(lab[crop_x-8:crop_x+8, crop_y-128:crop_y+128, crop_z-128:crop_z+128])
    
        prediction = test_single_volume(image, label, net, num_classes=2, patch_size=[256, 256])
        #print('prediction type ', type(prediction))
        prediction_np = prediction.numpy()
        #print('prediction_np type = ', type(prediction_np))
        pred_image = sitk.GetImageFromArray(prediction_np.astype(np.uint8))
        #print('image_path[1].split(''/'')[-1] type = ', type(image_path[1].split('/')[-1]))
        #print('222csvname type = ', type(csvname))
        
        #print('prediction shape = ',prediction.shape) #(355, 372, 372)
        
        #id = image_path[1].split('/')[-1]
        #iter_num = iter_num
        #metric = cal_metric(label,prediction)
        metric = cal_sparse(label,prediction)
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
        
    total_metric = total_metric[1:]
    avg_metric = np.around(np.mean(total_metric,axis=0),4) #averaged of dice, recall and precision
    cur_dice = avg_metric[0]

    print("Validation end")
    return avg_metric