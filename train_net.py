import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np

import time
import random
import os
import sys
from sklearn.metrics import  precision_recall_fscore_support,confusion_matrix, ConfusionMatrixDisplay


from Net import BasenetFgnnMeanfield
from data.build_dataset import build_dataset
from utils import *
from sklearn.metrics import plot_confusion_matrix


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def adjust_lr(optimizer, new_lr):
    print('change learning rate:', new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def createLogpath(cfg):
    # eg CAGNet/log/bit/line_name
    log_path=os.path.join(cfg.log_path,cfg.dataset_name,cfg.linename)
    if os.path.exists(log_path)==False:
        os.makedirs(log_path,exist_ok=True)
    return log_path

def train_net(cfg):
    """
    training net
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device_list

    log_path=createLogpath(cfg)
    print(log_path)

    # Reading dataset
    training_loader,validation_loader=build_dataset(cfg)
    # for iterr, i in enumerate((training_loader)):
    #     print(i.size())
    # exit()


    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        
    else:
        device = torch.device('cpu')

    model=BasenetFgnnMeanfield(cfg)
    model.to(device)

    # if cfg.use_multi_gpu:
    #     model = nn.DataParallel(model)

    # model = model.cuda()
    # model.train()  # train mode
    # model.apply(set_bn_eval)

    # if cfg.solver == 'adam':
    #     optimizer = optim.Adam(filter(lambda p: p.requires_grad,
    #                                   model.parameters()),
    #                            lr=cfg.train_learning_rate,
    #                            weight_decay=cfg.weight_decay)

    # start_epoch = 1

    if cfg.test==True:
        print('begin test')
        model.load_state_dict(torch.load(cfg.savedmodel_path), strict=False)
        if cfg.use_multi_gpu:
            model = nn.DataParallel(model)
        model = model.cuda()
        test_info = test_func(validation_loader, model, device, 'test', cfg)
        save_log(cfg,log_path,'test_info', test_info)
        print('end test')
        exit(0)


    if cfg.use_multi_gpu:
        model = nn.DataParallel(model)

    model = model.cuda()
    model.train()  # train mode
    model.apply(set_bn_eval)

    if cfg.solver == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      model.parameters()),
                               lr=cfg.train_learning_rate,
                               weight_decay=cfg.weight_decay)

    start_epoch = 1

    for epoch in range(start_epoch, start_epoch + cfg.max_epoch):
        train_func(training_loader, model, device, optimizer, epoch, cfg)
        
        if epoch % cfg.test_interval_epoch == 0:# evaluation
            test_info = test_func(validation_loader, model, device, epoch, cfg)
            save_log(cfg,log_path, 'log', test_info)
            filepath=os.path.join(log_path,f'epoch{epoch}.pth')
            if cfg.use_multi_gpu:
                torch.save(model.module.state_dict(),filepath)
            else:
                torch.save(model.state_dict(),filepath)
            print('*epoch ' + str(epoch) + ' finished')


def train_func(data_loader, model, device, optimizer, epoch, cfg):
    loss_meter = AverageMeter()
   
    for batch_data in data_loader:
        # print(batch_data) #it has the length of 7 as the number of variables that the Dataset class returns
        # exit()
        model.train()
        model.apply(set_bn_eval)
        batch_data_full=batch_data
        

        # prepare batch data
        batch_data = [b.to(device=device) for b in batch_data[:5]]  # move data to gpu      --get images, bboxes, actions, bboxes_num, interactions informations, seq_name, fid, normalized bboxes
        
        batch_size = batch_data[0].shape[0]
        # print(batch_size)
        num_frames = batch_data[0].shape[1]
        # print(num_frames) = 1
        

        # forward
        actions_scores, interaction_scores = model((batch_data[0], batch_data[1], batch_data[3]), batch_data_full[5], batch_data_full[6].to(device=device), batch_data_full[7].to(device=device))
        # print(actions_scores.shape)
        # print(interaction_scores.shape)
        # actions_scores has size of (bboxes_num, num_classes)
        # interaction_scores has size of (num_paired_people, 2)

        actions_in = batch_data[2].reshape((batch_size, num_frames, cfg.num_boxes))  # aligned action_label,which contains -1
        interactions_in = batch_data[4].reshape((batch_size, num_frames, cfg.num_boxes * (cfg.num_boxes - 1)))
        bboxes_num = batch_data[3].reshape(batch_size, num_frames)  # bbox_num

        actions_in_nopad = []
        interactions_in_nopad = []

        actions_in = actions_in.reshape((batch_size * num_frames, cfg.num_boxes,))
        interactions_in = interactions_in.reshape((batch_size * num_frames, cfg.num_boxes * (cfg.num_boxes - 1),))
        bboxes_num = bboxes_num.reshape(batch_size * num_frames, )
        for bt in range(batch_size * num_frames):
            N = bboxes_num[bt]
            actions_in_nopad.append(actions_in[bt, :N])
            interactions_in_nopad.append(interactions_in[bt, :N * (N - 1)])

        actions_in = torch.cat(actions_in_nopad, dim=0).reshape(-1, )  # ALL_N, sum of all valid action ground truth label
        interactions_in = torch.cat(interactions_in_nopad, dim=0).reshape(-1, )

        aweight,iweight=None,None
        if cfg.action_weight!=None:
            aweight=torch.tensor(cfg.action_weight,
                                 dtype=torch.float,
                                 device='cuda')
        if cfg.inter_weight!=None:
            iweight=torch.tensor(cfg.inter_weight,
                                 dtype=torch.float,
                                 device='cuda')
        # Predict actions
        actions_loss = F.cross_entropy(actions_scores,
                                       actions_in,
                                       weight=aweight)  # cross_entropy

        interactions_loss = F.cross_entropy(interaction_scores,
                                            interactions_in,
                                            weight=iweight)

        total_loss = actions_loss + interactions_loss

        loss_meter.update(val=total_loss.item(), n=batch_size)

        # Optim
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print('epoch: ' + str(epoch) + ', loss: ' + str(loss_meter.avg))
    # if cfg.use_multi_gpu:
    #     print('lambda_h:{},lambda_g:{}'.format(model.module.lambda_h.item(), model.module.lambda_g.item()))
    # else:
    #     print('lambda_h:{},lambda_g:{}'.format(model.lambda_h.item(), model.lambda_g.item()))


def test_func(data_loader, model, device, epoch, cfg):
    model.eval()

    actions_meter = AverageMeter() # AverageMeter() is a utility class in PyTorch that can be used to keep track of the average value of a given quantity during training 
    interactions_meter = AverageMeter()
    loss_meter = AverageMeter()
    actions_classification_labels = [0 for i in range(cfg.num_actions)]
    actions_classification_pred_true = [0 for i in range(cfg.num_actions)]
    interactions_classification_labels = [0, 0]
    interactions_classification_pred_true = [0, 0]

    actions_pred_global = []
    actions_labels_global = []
    interactions_pred_global = []
    interactions_labels_global = []

    with torch.no_grad():
        for batch_data in data_loader:
            # prepare batch data
            seq_name, fid = batch_data[-3], batch_data[-2]
            batch_data_full = batch_data
            
            batch_data = [b.to(device=device) for b in batch_data[0:5]]    #--get images, bboxes, actions, bboxes_num, interactions informations
            batch_size = batch_data[0].shape[0]
            num_frames = batch_data[0].shape[1]
            
            # print(batch_data[0].shape)    [32, 1, 3, 540, 960] -- batch_size, num_frames, channels, height, width
            # print(batch_data[1].shape)    [32, 1, 15, 4] -- batch_size, num_frames, num_boxes, coordinates of each bbox. The bboxes is aligned to the maximum num_boxes = 15 with (0, 0, 0, 0)
            # print(batch_data[2].shape)    [32, 1, 15] -- batch_size, num_frames, max_num_boxes. For aligned bboxes (0, 0, 0, 0), the action labels will be -1 
            # print(batch_data[3].shape)    [32, 1] -- batch_size, num_bboxes (num_people)
            # print(batch_data[4].shape)    [32, 1, 210] -- batch_size, num_frames, 15*14
            # print("------------------------------------------------------")
            
            actions_in = batch_data[2].reshape((batch_size, num_frames, cfg.num_boxes))
            interactions_in = batch_data[4].reshape((batch_size, num_frames, cfg.num_boxes * (cfg.num_boxes - 1)))

            bboxes_num = batch_data[3].reshape(batch_size, num_frames)

            # forward
            actions_scores, interactions_scores = model((batch_data[0], batch_data[1], batch_data[3]), batch_data_full[5], batch_data_full[6].to(device=device), batch_data_full[7].to(device=device))
            # print(actions_scores.shape) (number_people, num_classes)
            # print(interactions_scores.shape) (number_paired_people, 2)

            actions_in_nopad = []
            interactions_in_nopad = []

            actions_in = actions_in.reshape((batch_size * num_frames, cfg.num_boxes,))
            interactions_in = interactions_in.reshape(
                (batch_size * num_frames, cfg.num_boxes * (cfg.num_boxes - 1),))
            bboxes_num = bboxes_num.reshape(batch_size * num_frames, )
            for bt in range(batch_size * num_frames):
                N = bboxes_num[bt] # get num_bboxes (people) of each frame
                actions_in_nopad.append(actions_in[bt, :N]) # eliminate padding of bboxes
                interactions_in_nopad.append(interactions_in[bt, :N * (N - 1)]) # eliminate padding

            actions_in = torch.cat(actions_in_nopad, dim=0).reshape(-1, )  # ALL_N,
            interactions_in = torch.cat(interactions_in_nopad, dim=0).reshape(-1, )
            # For example, actions_in has the size of torch.Size([80]) which is the total number of all bboxes (people) in all sample frames in the batch 
            # print(actions_in.shape)   
            # print(interactions_in.shape)

            aweight, iweight = None, None
            if cfg.action_weight != None:
                aweight = torch.tensor(cfg.action_weight,
                                       dtype=torch.float,
                                       device='cuda')
            if cfg.inter_weight != None:
                iweight = torch.tensor(cfg.inter_weight,
                                       dtype=torch.float,
                                       device='cuda')

            actions_loss = F.cross_entropy(actions_scores,
                                           actions_in,
                                           weight=aweight)

            interactions_loss = F.cross_entropy(interactions_scores,
                                                interactions_in,
                                                weight=iweight)

            actions_pred = torch.argmax(actions_scores, dim=1)  # ALL_N,
            actions_correct = torch.sum(torch.eq(actions_pred.int(), actions_in.int()).float())

            interactions_pred = torch.argmax(interactions_scores, dim=1)
            interactions_correct = torch.sum(torch.eq(interactions_pred.int(), interactions_in.int()).float())
            
            
            # Get all action prediction and ground truth
            actions_pred_global.append(actions_pred.cpu())
            actions_labels_global.append(actions_in.cpu())
            interactions_pred_global.append(interactions_pred.cpu())
            interactions_labels_global.append(interactions_in.cpu())
            
            # Print results 
            # print("seq_name: ", seq_name)
            # print("fid: ", fid)
            # print("actions_pred: ", actions_pred)
            # print("actions_in: ", actions_in)
            # print("interactions_pred: ", interactions_pred)
            # print("interactions_in: ", interactions_in)

            # calculate recall
            for i in range(len(actions_pred)):
                actions_classification_labels[actions_in[i]] += 1
                if actions_pred[i] == actions_in[i]:
                    actions_classification_pred_true[actions_pred[i]] += 1
            for i in range(len(interactions_pred)):
                interactions_classification_labels[interactions_in[i]] += 1
                if interactions_pred[i] == interactions_in[i]:
                    interactions_classification_pred_true[interactions_pred[i]] += 1
            # Get accuracy
            actions_accuracy = \
                actions_correct.item() / actions_scores.shape[0]
            interactions_accuracy = \
                interactions_correct.item() / interactions_in.shape[0]

            actions_meter.update(actions_accuracy, actions_scores.shape[0])
            interactions_meter.update(interactions_accuracy, interactions_in.shape[0])

            # Total loss
            total_loss = actions_loss + interactions_loss
            loss_meter.update(total_loss.item(), batch_size)

    for i in range(len(actions_classification_labels)):
        actions_classification_pred_true[i] = \
            actions_classification_pred_true[i] * 1.0 / actions_classification_labels[i]
    for i in range(len(interactions_classification_labels)):
        interactions_classification_pred_true[i] = \
            interactions_classification_pred_true[i] * 1.0 / interactions_classification_labels[i]

    actions_pred_global = torch.cat(actions_pred_global)
    actions_labels_global = torch.cat(actions_labels_global)
    interactions_pred_global = torch.cat(interactions_pred_global)
    interactions_labels_global = torch.cat(interactions_labels_global)

    # calculate mean IOU for each class, then average
    cls_iou = torch.Tensor([0 for _ in range(cfg.num_actions + 2)]).cuda().float()
    for i in range(cfg.num_actions):
        # Find indices of samples that have label of i in the list actions_labels_global
        grd = set((actions_labels_global == i).nonzero().squeeze(dim=-1).cpu().numpy().tolist())    #nonzero() returns the indices of the elements in a tensor that are non-zero 
        prd = set((actions_pred_global == i).nonzero().squeeze(dim=-1).cpu().numpy().tolist())
        uset = grd.union(prd)       # union() combines two sets into a single set that contains all the unique elements from both sets
        iset = grd.intersection(prd)    # intersection() finds the common elements between two sets 
        cls_iou[i] = len(iset) / len(uset)  # iou for each class
    for i in range(2):
        grd = set((interactions_labels_global == i).nonzero().squeeze(dim=-1).cpu().numpy().tolist())
        prd = set((interactions_pred_global == i).nonzero().squeeze(dim=-1).cpu().numpy().tolist())
        uset = grd.union(prd)
        iset = grd.intersection(prd)
        cls_iou[cfg.num_actions + i] = len(iset) / len(uset)    # iou for interaction/non-interaction
    mean_iou = cls_iou.mean()

    actions_precision, actions_recall, actions_F1, support = precision_recall_fscore_support(actions_labels_global,
                                                                                             actions_pred_global,
                                                                                             beta=1, average='macro')
    interactions_precision, interactions_recall, interactions_F1, support = precision_recall_fscore_support(
        interactions_labels_global, interactions_pred_global, beta=1, average='macro')

    
    # conf_mat=confusion_matrix(interactions_labels_global.cpu().numpy(),interactions_pred_global.cpu().numpy())
    # conf_mat = conf_mat/np.expand_dims(np.sum(conf_mat, axis=1), axis=1)
    # conf_mat = np.round(conf_mat, 2)
    # conf_mat[conf_mat == 0.00] = 0
    
    # disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    # import matplotlib.pyplot as plt
    # # Set the font size
    # plt.rcParams.update({'font.size': 35})
    # # Plot the confusion matrix
    # fig, ax = plt.subplots(figsize=(30, 30))  # Create a new figure with a custom size
    # disp.plot(include_values=True, cmap='Blues', ax=ax, xticks_rotation='horizontal')
    
    # # Change the size of the tick parameters
    # ax.tick_params(axis='both', which='major', labelsize=35)
    # # Set the labels
    # ax.set_xlabel('Predicted Label', fontsize=35)
    # ax.set_ylabel('True Label', fontsize=35)
    # plt.show()
    

    test_info = {
        # 'time':epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        # 'activities_acc':activities_meter.avg*100,
        'actions_precision': actions_precision,
        'actions_recall': actions_recall,
        'actions_F1': actions_F1,
        'actions_acc': actions_meter.avg * 100,
        'actions_classification_recalls': actions_classification_pred_true,
        'interactions_precision': interactions_precision,
        'interactions_recall': interactions_recall,
        'interactions_F1': interactions_F1,
        'interactions_acc': interactions_meter.avg * 100,
        'interactions_classification_recalls': interactions_classification_pred_true,
        'mean_iou': mean_iou
        # 'confusion_matrix':conf_mat
    }

    return test_info
