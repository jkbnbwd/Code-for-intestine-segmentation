# -*- coding: utf-8 -*-
# Author: Xiangde Luo
# Date:   16 Dec. 2021
# Implementation for Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer.
# # Reference:
#   @article{luo2021ctbct,
#   title={Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer},
#   author={Luo, Xiangde and Hu, Minhao and Song, Tao and Wang, Guotai and Zhang, Shaoting},
#   journal={arXiv preprint arXiv:2112.04894},
#   year={2021}}
#   In the original paper, we don't use the validation set to select checkpoints and use the last iteration to inference for all methods.
#   In addition, we combine the validation set and test set to report the results.
#   We found that the random data split has some bias (the validation set is very tough and the test set is very easy).
#   Actually, this setting is also a fair comparison.
#   download pre-trained model to "code/pretrained_ckpt" folder, link:https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY

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
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from config import get_config
from dataloaders import utils
#from dataloaders.dataset import (BaseDataSets, RandomGenerator,
#                                 TwoStreamBatchSampler)
from dataloaders.brats2019 import (BraTS2019, Intestine, Intestine_trans,CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,Cutout, Normalization,
                                   TwoStreamBatchSampler, OneStreamBatchSampler)
from networks.net_factory import net_factory
from networks.net_factory_3d import net_factory_3d
from networks.vision_transformer import SwinUnet as ViT_seg
from utils import losses, metrics, ramps
from val_2D import test_all_case_2d
from val_3D_pro import test_all_case
import torch.nn.functional as F
import pdb # pdb.set_trace()
import pandas as pd
import csv
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/dataT1/Free/qinan/intestine/train/mul-semi', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='test', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='ri2', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=50000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=int,  default=5, help='GPU number')
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
parser.add_argument('--labeled_bs', type=int, default=1,
                    help='labeled_batch_size per gpu')
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
#pdb.set_trace()

os.environ["CUDA_VISIBLE_DEVICES"] = str(5)

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model



def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    train_data_path = '/dataT1/Free/qinan/intestine/train/bigpatch/train_path_list.txt'
    val_data_path = '/dataT1/Free/qinan/intestine/train/bigpatch/val_path_list.txt'

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    trans = ViT_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes).cuda()
    trans.load_from(config)
    #net2 = create_model()

    net2 = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes).cuda() 
    model2 = kaiming_normal_init_weight(net2)

    net2 = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes).cuda() 
    model2 = kaiming_normal_init_weight(net2)
    #ckpt = torch.load('./featuremap/best_model1.pth')
    #model2.load_state_dict(ckpt)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train_3d = Intestine_trans(base_dir=train_data_path,
                            num=None,
                            transform=transforms.Compose([
                                RandomRotFlip(),
                                RandomCrop(args.patch_size),
                                ToTensor(),
                                #Normalization(),
                                Cutout(),
                            ]))
    #
    #'''
    labeled_slice = 150 #1664
    total_slices = 176#4662
    #'''
    
    '''#3 labeled cases
    labeled_slice = 440
    total_slices = 3438
    '''
    
    '''#9 labeled cases
    labeled_slice = 1164
    total_slices = 4162
    '''

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    #trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
    #                         num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    trainloader_3d = DataLoader(db_train_3d, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    trans.train()
    model2.train()

    optimizer1 = optim.SGD(trans.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader_3d)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader_3d) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader_3d):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            
            B = args.patch_size[0]*args.batch_size
            #print('B = ', B)
            #1/0
            ps1, ps2, ps3 = args.patch_size # size of patch
            #print('patch size shape = ',ps1, ps2, ps3 )
            #1/0


            volume_batch_2d = torch.reshape(volume_batch, [B,1,ps2,ps3])

            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            volume_batch_2d = volume_batch_2d.cuda()
            consistency_weight = get_current_consistency_weight(iter_num // 150)         

            #outputs1 = trans(volume_batch_2d) 
                 
            outputs2 = model2(volume_batch)
         
        
            outputs1 = torch.reshape(outputs1, (outputs2.shape))

            outputs_soft1 = torch.softmax(outputs1, dim=1) 
            outputs_soft2 = torch.softmax(outputs2, dim=1)


            ce1 = ce_loss(outputs1[:args.labeled_bs], label_batch[:args.labeled_bs].long())
            dice1 = dice_loss(outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss1 = ce1 + dice1

            ce2 = ce_loss(outputs2[:args.labeled_bs], label_batch[:args.labeled_bs].long())
            dice2 = dice_loss(outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss2 = ce2 + dice2
            
            #pseudo_supervision1 = dice_loss(outputs1[args.labeled_bs:], outputs2[args.labeled_bs:])
            #pseudo_supervision2 = dice_loss(outputs2[args.labeled_bs:], outputs1[args.labeled_bs:])

            pseudo_outputs1 = torch.argmax(
                outputs_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(
                outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)
            #print('pseudo_outputs1 shape = ', pseudo_outputs1.shape)
            #print('outputs_soft1 shape = ', outputs_soft1.shape)

            pseudo_supervision1 = dice_loss(outputs_soft1[args.labeled_bs:], pseudo_outputs2.unsqueeze(1))
            pseudo_supervision2 = dice_loss(outputs_soft2[args.labeled_bs:], pseudo_outputs1.unsqueeze(1))
            
            trans_loss =  loss1 + pseudo_supervision1 * consistency_weight
            model2_loss = loss2 + pseudo_supervision2 * consistency_weight

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss = 0.5*(trans_loss + model2_loss)

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
           
            writer.add_scalar('loss/trans_loss',
                              loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              loss, iter_num)
       
            logging.info('iteration %d : trans loss : %f model2 loss : %f ' % (
                iter_num, trans_loss.item(),model2_loss.item()))
            
            col_name = ['iter_num','dice','recall','precision']
            if iter_num > 0 and iter_num % 200 == 0:
                trans.eval()
                metric_list1 = 0.0
                metric_list1 = test_all_case_2d(args.exp,
                trans, val_data_path, num_classes=2, patch_size=args.patch_size,
                stride_xy=128, stride_z=8, tar = 'trans')
                metric_list1 =  np.insert(metric_list1,0,iter_num)
                metric_list1 = np.array(metric_list1).reshape(1,4)

                test =  pd.DataFrame(columns = col_name, data=metric_list1)
                if not os.path.exists('../csv/'+args.exp+'/'):
                    os.makedirs('../csv/'+args.exp+'/')
                    test.to_csv('../csv/'+args.exp+'/'+'trans_val.csv',encoding='gbk',mode = 'a')
                else:
                    test.to_csv('../csv/'+args.exp+'/'+'trans_val.csv',encoding='gbk',mode = 'a',header=False)


                if metric_list1[0,1] > best_performance1:
                    best_performance1 = metric_list1[0,1]
                    save_mode_path = os.path.join(snapshot_path,
                                                  'trans_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    if not os.path.exists(snapshot_path+'/best_model/'):
                        os.makedirs(snapshot_path+'/best_model/')
                    save_best = os.path.join(snapshot_path+'/best_model',
                                             'best_model1.pth')
                    torch.save(trans.state_dict(), save_mode_path)
                    torch.save(trans.state_dict(), save_best)
                
                logging.info(
                    'iteration %d : trans_mean_dice : %f' % (iter_num, metric_list1[0,1]))
                trans.train()
                model2.eval()
                
                metric_list2 = test_all_case(args.exp,
                model2, val_data_path, num_classes=2, patch_size=[16,256,256],
                stride_xy=128, stride_z=8, tar = 'unet')
                metric_list2 =  np.insert(metric_list2,0,iter_num)
                metric_list2 = np.array(metric_list2).reshape(1,4)
                test =  pd.DataFrame(columns = col_name, data=metric_list2)
                if not os.path.exists('../csv/'+args.exp+'/'):
                    os.makedirs('../csv/'+args.exp+'/')
                    test.to_csv('../csv/'+args.exp+'/'+'unet_val.csv',encoding='gbk',mode = 'a')
                else:
                    test.to_csv('../csv/'+args.exp+'/'+'unet_val.csv',encoding='gbk',mode = 'a',header=False)
                
                if metric_list2[0,1] > best_performance2:
                    trigger2 = 0
                    best_performance2 = metric_list2[0,1]
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2, 4)))
                    
                    save_best = os.path.join(snapshot_path+'/best_model','best_model2.pth')
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)


                writer.add_scalar('info/model2_val_dice',metric_list2[0, 1], iter_num)

                writer.add_scalar('info/model2_val_mean_dice',
                                  metric_list2[0,1], iter_num)
                logging.info(
                    'iteration %d : model2_mean_dice : %f' % (iter_num, metric_list2[0,1]))
                model2.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'trans_iter_' + str(iter_num) + '.pth')
                torch.save(trans.state_dict(), save_mode_path)
                logging.info("save trans to {}".format(save_mode_path))
                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
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

    snapshot_path = "../model/{}".format(args.exp)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))
    
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
