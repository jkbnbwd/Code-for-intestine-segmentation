import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import SimpleITK as sitk
#import cv2
from utils.losses import soft_cldice,soft_dice_cldice

from dataloaders import utils
from dataloaders.brats2019 import (BraTS2019, Intestine, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,Cutout, Normalization,
                                   TwoStreamBatchSampler)
from networks.net_factory_3d import net_factory_3d #choose model type
from utils import losses, metrics, ramps
from val_3D import test_all_case
#from gradcam import weightmap
from scipy import ndimage
import pandas as pd
import csv
#from torch import summary
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/dataT1/Free/qinan/intestine/train', help='Path of Dataset') #default='../data/BraTS2019'
parser.add_argument('--csvname', type=str,
                    default='data', help='experiment_name')
parser.add_argument('--data_name', type=str,
                    default='inte', help='experiment_name')
parser.add_argument('--loss', type=str,
                    default='dice', help='experiment_name')
parser.add_argument('--sup', type=str,
                    default='cl', help='experiment_name')
parser.add_argument('--dis', type=int,
                    default=0, help='experiment_name')
parser.add_argument('--model1', type=str,
                    default='ri2', help='model_name')
parser.add_argument('--model2', type=str,
                    default='ri2', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--val_num', type=int,
                    default=200, help='maximum number to val')
parser.add_argument('--batch', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[16,256,256],#default=[96, 96, 96],
                    help='patch size of network input')
parser.add_argument('--pretrain', type=int,default=0, help='use pretrain')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=int,default=0, help='use cpu only')
parser.add_argument('--early_stop', type=int,default=30, help='early stop strategy')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=4,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=548, #3
                    help='labeled data')

# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()

gpu = str(args.gpu)

os.environ["CUDA_VISIBLE_DEVICES"] = gpu

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def kaiming_normal_init_weight(model):
    for m in model.modules(): #model.modules() model's layers
        if isinstance(m, nn.Conv3d): #判断一个对象是否是一个已知的类型
            torch.nn.init.kaiming_normal_(m.weight) #initial weight
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1) #initial weight
            m.bias.data.zero_()    #initial weight
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight) #initial weight
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1) #initial weight
            m.bias.data.zero_()  #initial weight
    return model

try:
    from cupyx.scipy.ndimage import distance_transform_edt
except Exception as exc:
    print(exc)
    print("Using CPU distance transform")
    from scipy.ndimage import distance_transform_edt

def weighted_dice(pred,pseudogt):
    smooth = 1.0
    # If cupyx is available, we can work directly with GPU tensors
    if 'cupyx' in globals():
        # Cupy operates on GPU arrays, so we must convert the pseudogt to cupy array
        import cupy as cp
        
        # Ensure pseudogt is on the GPU, convert to cupy array
        pseudogt_cp = cp.asarray(pseudogt)
        
        # Perform distance transform directly on the GPU
        dis_map = distance_transform_edt(pseudogt_cp)  # works on cupy array on GPU
    else:
        # Use scipy on CPU (convert pseudogt to numpy array on CPU)
        dis_map = distance_transform_edt(pseudogt.cpu().numpy())

    # normalization
    # Normalize the distance map
    dis_map_mean = dis_map.mean()
    dis_map_std = dis_map.std()

    # Avoid division by zero
    dis_map_normalized = (dis_map - dis_map_mean) / (dis_map_std + 1e-6)

    dis_map_torch = torch.tensor(dis_map_normalized, dtype=torch.float32, device=pseudogt.device)
    dw_pseudo = pseudogt*dis_map_torch
    inter = pred*dw_pseudo
    dicescore = 2.0 * (inter.sum()+smooth) / (pred.sum()+dw_pseudo.sum()+smooth)
    lossvalue = 1.0 - dicescore
    return lossvalue
    

def train(args, snapshot_path):

    #print('snapshot_path =  ', snapshot_path)
    #1/0
    
    base_lr = args.base_lr
    train_data_path = args.root_path
    
    batch_size = args.batch
    max_iterations = args.max_iterations
    num_classes = 2
    early_stop = args.early_stop
    train_data_path = '/dataT1/Free/qinan/intestine/train/train_path_list.txt'
    val_data_path = '/dataT1/Free/qinan/intestine/train/val_path_list.txt'
    
 
    net1 = net_factory_3d(net_type=args.model1, in_chns=1, class_num=num_classes).cuda()
    model1 = kaiming_normal_init_weight(net1)
    
    net2 = net_factory_3d(net_type=args.model2, in_chns=1, class_num=num_classes).cuda() #network type

    model2 = xavier_normal_init_weight(net2)
    model1.train()
    model2.train()

    #loading dataset

    db_train = Intestine(base_dir=train_data_path,
                            num=None,
                            transform=transforms.Compose([
                                RandomRotFlip(),
                                RandomCrop(args.patch_size),
                                ToTensor(),
                                #Normalization(),
                                Cutout(),
                            ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    
    labeled_idxs = list(range(0, 1664)) #1664
    unlabeled_idxs = list(range(1664, 4662))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)
    

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    best_performance1 = 0.0
    best_performance2 = 0.0
    a1 = 0.8
    iter_num = 0
    if args.dis:
        ce_loss = CrossEntropyLoss(reduction='none')
    else:
        ce_loss = CrossEntropyLoss()

    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = max_iterations // len(trainloader) + 1

    iterator = tqdm(range(max_epoch), ncols=70)
    trigger1 = 0
    trigger2 = 0
    for epoch_num in iterator:
        stop_flag = 0
        #print('epoch_num  = ', epoch_num)
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()


            outputs1 = model1(volume_batch)
            
            outputs_soft1 = torch.softmax(outputs1, dim=1)

            outputs2 = model2(volume_batch)
           
            outputs_soft2 = torch.softmax(outputs2, dim=1)
       
            consistency_weight = get_current_consistency_weight(iter_num // 150)


            clloss =  soft_cldice()
            clloss1 = clloss(outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            clloss2 = clloss(outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))

            if clloss1 < 0 or clloss2 < 0:
                print('clloss1 = ', clloss1)
                print('clloss2 = ', clloss2)
                print("loss <<<<< 0")
                1/0
            if args.sup == 'cl':
                loss1 = 0.5 * (a1*dice_loss(outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)) \
                                            + (1-a1)*clloss1)
                loss2 = 0.5 * (a1*dice_loss(outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)) + \
                                (1-a1)*clloss2)

            if args.sup == 'ce':
                loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs],label_batch[:][:args.labeled_bs].long()))
                loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs],label_batch[:][:args.labeled_bs].long()))
            
            if args.sup == 'dice':
                loss1 = 0.5 * (dice_loss(outputs_soft1[:args.labeled_bs],label_batch[:args.labeled_bs].unsqueeze(1)))
                loss2 = 0.5 * (dice_loss(outputs_soft1[:args.labeled_bs],label_batch[:args.labeled_bs].unsqueeze(1)))   
            
            #hard label
            pseudo_outputs1 = torch.argmax(outputs_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)
    
   
            if args.loss == 'dice':
                pseudo_supervision1 = weighted_dice(outputs_soft1[args.labeled_bs:, 1, :], pseudo_outputs2)
                pseudo_supervision2 = weighted_dice(outputs_soft1[args.labeled_bs:, 1, :], pseudo_outputs1)
                
            
            else:
                pseudo_supervision1 = ce_loss(outputs1[args.labeled_bs:], pseudo_outputs2)
                pseudo_supervision2 = ce_loss(outputs2[args.labeled_bs:], pseudo_outputs1)

            model1_loss = loss1 + consistency_weight * pseudo_supervision1
            model2_loss = loss2 + consistency_weight * pseudo_supervision2
            loss = model1_loss + model2_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group1 in optimizer1.param_groups:
                param_group1['lr'] = lr_
            for param_group2 in optimizer2.param_groups:
                param_group2['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            logging.info(
                'iteration %d : model1 loss : %f model2 loss : %f' % (iter_num, model1_loss.item(), model2_loss.item()))
            #1/0

            col_name = ['iter_num','dice','recall','precision']
            if iter_num > 0 and iter_num % args.val_num == 0: #200
                model1.eval()
                trigger1 += 1
                avg_metric1 = test_all_case(args.csvname,
                    model1, args.root_path, test_list="val_path_list.txt", num_classes=2, patch_size=args.patch_size,
                    stride_xy=128, stride_z=8, tar = 'model1')
                
                avg_metric1 =  np.insert(avg_metric1,0,iter_num)
                #print('avg_metric1 shape =',avg_metric1.shape)
                avg_metric1 = np.array(avg_metric1).reshape(1,4)
                
                test =  pd.DataFrame(columns = col_name, data=avg_metric1)
                if not os.path.exists('../csv/'+args.csvname+'/'):
                    os.makedirs('../csv/'+args.csvname+'/')
                    test.to_csv('../csv/'+args.csvname+'/'+'model1_val.csv',encoding='gbk',mode = 'a')
                else:
                    test.to_csv('../csv/'+args.csvname+'/'+'model1_val.csv',encoding='gbk',mode = 'a',header=False)
                trigger1 += 1
                if avg_metric1[0,1] >= best_performance1:
                    trigger1 = 0
                    best_performance1 = avg_metric1[0,1]
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    if not os.path.exists(snapshot_path+'/best_model/'):
                        os.makedirs(snapshot_path+'/best_model/')
                    save_best = os.path.join(snapshot_path+'/best_model',
                                             'best_model1.pth')
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                writer.add_scalar('info/model1_val_dice_score',
                                  avg_metric1[0,1], iter_num)

                logging.info(
                    'iteration %d : model1_dice_score : %f' % (
                        iter_num, avg_metric1[0,1]))
                        
                model1.train()
                model2.eval()
                
                avg_metric2 = test_all_case(args.csvname, 
                    model2, args.root_path, test_list="val_path_list.txt", num_classes=2, patch_size=args.patch_size,
                    stride_xy=128, stride_z=8,tar = 'model2') #64 64
                 
                avg_metric2 =  np.insert(avg_metric2,0,iter_num)
                avg_metric2 = np.array(avg_metric2).reshape(1,4)
                #col_name = ['iter_num','dice','recall','precision']
                test =  pd.DataFrame(columns = col_name, data=avg_metric2)
                if not os.path.exists('../csv/'+args.csvname+'/'):
                    os.makedirs('../csv/'+args.csvname+'/')
                    test.to_csv('../csv/'+args.csvname+'/model2_val.csv',encoding='gbk',mode = 'a')
                else:
                    test.to_csv('../csv/'+args.csvname+'/model2_val.csv',encoding='gbk',mode = 'a',header=False)
                trigger2  += 1
                if avg_metric2[0,1] > best_performance2:
                    trigger2 = 0
                    best_performance2 = avg_metric2[0,1]
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path+'/best_model',
                                             'best_model2.pth')
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)
                
                
                writer.add_scalar('info/model2_val_dice_score',
                                  avg_metric2[0,1], iter_num)
                #writer.add_scalar('info/model2_val_hd95',
                #                  avg_metric2[0, 1], iter_num)
                logging.info(
                    'iteration %d : model2_dice_score : %f' % (
                        iter_num, avg_metric2[0,1]))
                model2.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
            if early_stop > 0:
                
                if trigger1 >= early_stop & trigger2 >= early_stop:
                    #print("=> inner stopping")
                    stop_flag = 1
                    break
                
        if stop_flag :
            print("=> early stopping")
            break
        stop_flag = 0
        if iter_num >= max_iterations:
            iterator.close()
            break
        #print('early_stop = ', args.early_stop)    
        
    writer.close()
    

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    snapshot_path = "../model/{}".format(
        args.csvname) #, args.model1+args.model2) #args.exp
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)


    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)

    
