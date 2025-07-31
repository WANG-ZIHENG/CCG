import os
import numpy as np
import random
import argparse
import time
import json
import pandas as pd
from collections import Counter
# from geomloss import SamplesLoss


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

import sys
sys.path.append('.')

from cycle_resnet_src.modules import *
from cycle_resnet_src import logger


parser = argparse.ArgumentParser(description='FairCLIP Training/Fine-Tuning')

parser.add_argument('--seed', default=-1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--num_epochs', default=90, type=int)
parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=6e-5, type=float,
                    metavar='W', help='weight decay (default: 6e-5)',
                    dest='weight_decay')

parser.add_argument('--result_dir', default='./results/results', type=str)
parser.add_argument('--dataset_dir', default='/root/data/fairvlmed10k', type=str)
# parser.add_argument('--dataset_dir', default='/H_share/data/fairvlmed10k', type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--workers', default=12, type=int)
parser.add_argument('--eval_set', default='test', type=str, help='options: val | test')
# parser.add_argument('--summarized_note_file', default='/H_share/data/fairvlmed10k/gpt-4_summarized_notes.csv', type=str)
parser.add_argument('--summarized_note_file', default='/root/data/fairvlmed10k/data_summary_all.csv',
                    type=str)
parser.add_argument('--text_source', default='note', type=str, help='options: note | label')
parser.add_argument('--perf_file', default='', type=str)
parser.add_argument('--model_arch', default='efficientnet_b4', type=str, help='options: efficientnet_b0 | efficientnet_b1 | efficientnet_b2 | efficientnet_b3 | efficientnet_b4 | efficientnet_b5 | efficientnet_b6 | efficientnet_b7')
parser.add_argument('--pretrained_weights', default='', type=str)
parser.add_argument('--use_gen_data', default=False, type=bool)



if __name__ == '__main__':
    args = parser.parse_args()

    if args.seed < 0:
        args.seed = int(np.random.randint(10000, size=1)[0])
    set_random_seed(args.seed)

    logger.log(f'===> random seed: {args.seed}')

    logger.configure(dir=args.result_dir, log_suffix='train')

    with open(os.path.join(args.result_dir, f'args_train.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # the number of groups in each attribute
    groups_in_attrs = [3, 2, 2, 3]
    attr_to_idx = {'race': 0, 'gender': 1, 'ethnicity': 2, 'language': 3}

    model_arch_mapping = {'vit-b16': 'ViT-B/16', 'vit-l14': 'ViT-L/14'}

    best_global_perf_file = os.path.join(os.path.dirname(args.result_dir), f'best_{args.perf_file}')
    acc_head_str = ''
    auc_head_str = ''
    dpd_head_str = ''
    eod_head_str = ''
    esacc_head_str = ''
    esauc_head_str = ''
    group_disparity_head_str = ''
    if args.perf_file != '':
        if not os.path.exists(best_global_perf_file):
            for i in range(len(groups_in_attrs)):
                auc_head_str += ', '.join([f'auc_attr{i}_group{x}' for x in range(groups_in_attrs[i])]) + ', '
            dpd_head_str += ', '.join([f'dpd_attr{x}' for x in range(len(groups_in_attrs))]) + ', '
            eod_head_str += ', '.join([f'eod_attr{x}' for x in range(len(groups_in_attrs))]) + ', '
            esacc_head_str += ', '.join([f'esacc_attr{x}' for x in range(len(groups_in_attrs))]) + ', '
            esauc_head_str += ', '.join([f'esauc_attr{x}' for x in range(len(groups_in_attrs))]) + ', '

            group_disparity_head_str += ', '.join([f'std_group_disparity_attr{x}, max_group_disparity_attr{x}' for x in range(len(groups_in_attrs))]) + ', '
             
            
            with open(best_global_perf_file, 'w') as f:
                f.write(f'epoch, acc, {esacc_head_str} auc, {esauc_head_str} {auc_head_str} {dpd_head_str} {eod_head_str} {group_disparity_head_str} path\n')

    device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    # 根据args.model_arch选择efficientnet架构
    efficientnet_archs = {
        'efficientnet_b0': models.efficientnet_b0,
        'efficientnet_b1': models.efficientnet_b1,
        'efficientnet_b2': models.efficientnet_b2,
        'efficientnet_b3': models.efficientnet_b3,
        'efficientnet_b4': models.efficientnet_b4,
        'efficientnet_b5': models.efficientnet_b5,
        'efficientnet_b6': models.efficientnet_b6,
        'efficientnet_b7': models.efficientnet_b7,
    }
    if args.model_arch in efficientnet_archs:
        efficientnet_b4_weights = models.EfficientNet_B4_Weights.DEFAULT
        model = efficientnet_archs[args.model_arch](weights=efficientnet_b4_weights)
        num_features = model.classifier[1].in_features
        num_classes = 1  # 假设是二分类任务
        model.classifier[1] = nn.Linear(num_features, num_classes)
    
    elif args.model_arch == 'resnet18':
        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        num_classes = 1  # 假设是二分类任务
        model.fc = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"暂不支持的模型架构: {args.model_arch}")
    
    model = model.to(device)

    train_files = 'filter_file.txt'
    test_files = None
    # 数据增强的transforms
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=512),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        # transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # 测试集的transforms
    transform_test = transforms.Compose([
        transforms.RandomResizedCrop(size=512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dataset = fair_vl_med_dataset(args,args.dataset_dir, transform_train,use_gen_data=args.use_gen_data, subset='training', text_source=args.text_source, summarized_note_file=args.summarized_note_file,files=train_files)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False,)

    val_dataset = fair_vl_med_dataset(args,args.dataset_dir, transform_test, summarized_note_file=args.summarized_note_file, subset='validation')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    test_dataset = fair_vl_med_dataset(args,args.dataset_dir, transform_test, summarized_note_file=args.summarized_note_file, subset='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)



    logger.log(f'# of training samples: {train_dataset.__len__()}, # of val samples: {val_dataset.__len__()}')



    group_size_on_race, group_size_on_gender, group_size_on_ethnicity = count_number_of_groups(train_dataset)
    logger.log(f'group size on race in training set: {group_size_on_race}')
    logger.log(f'group size on gender in training set: {group_size_on_gender}')
    logger.log(f'group size on ethnicity in training set: {group_size_on_ethnicity}')
    group_size_on_race, group_size_on_gender, group_size_on_ethnicity = count_number_of_groups(val_dataset)
    logger.log(f'group size on race in test set: {group_size_on_race}')
    logger.log(f'group size on gender in test set: {group_size_on_gender}')
    logger.log(f'group size on ethnicity in test set: {group_size_on_ethnicity}')


    bce = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.1, 0.1), eps=1e-6,weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    if args.pretrained_weights != "":
        checkpoint = torch.load(args.pretrained_weights)

        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    best_epoch = 0
    best_loss = 1000000
    best_pred_gt_by_attr = None
    best_auc = sys.float_info.min
    best_acc = sys.float_info.min
    best_es_acc = sys.float_info.min
    best_es_auc = sys.float_info.min


    for epoch in range(args.num_epochs):
        avg_loss = 0
        model.train()
        for batch in train_dataloader :


            optimizer.zero_grad()

            images, _, label_and_attributes,glaucoma_label,_ = batch

            images= images.to(device)
            # texts = texts.to(device)
            glaucoma_label = glaucoma_label.to(device)
            logits_per_image = model(images)
            logits_per_image = logits_per_image.squeeze()
            loss = bce(logits_per_image,glaucoma_label.float())
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)

            optimizer.step()



            avg_loss += loss.item()
            


        avg_loss /= len(train_dataloader)
        
        # iterate over val dataset
        model.eval()
        if epoch == args.num_epochs-1:
            loaders = [val_dataloader, test_dataloader]
        else:
            loaders = [val_dataloader]
        for loader in loaders:
            eval_avg_loss = 0
            all_probs = []
            all_labels = []
            all_attrs = []
            if loader == test_dataloader:
                logger.log(
                    f'==== test =====')
                
                model.load_state_dict(torch.load(os.path.join(args.result_dir, f"best.pth"), weights_only=False)['model_state_dict'])

            for batch in loader :
                images,_, label_and_attributes,glaucoma_label,_ = batch

                images= images.to(device)
                # texts = texts.to(device)
                glaucoma_label = glaucoma_label.to(device)
                glaucoma_labels = label_and_attributes[:, 0].to(device)
                attributes = label_and_attributes[:, 1:].to(device)

                class_text_feats = []
                with torch.no_grad():
                    image_features = model(images)
                    image_features = image_features.squeeze()

                vl_prob = torch.sigmoid(image_features)
                vl_logits = image_features

                all_probs.append(vl_prob.cpu().numpy())
                all_labels.append(glaucoma_labels.cpu().numpy())
                all_attrs.append(attributes.cpu().numpy())

                # apply binary cross entropy loss
                loss = bce(image_features, glaucoma_labels.float())
                eval_avg_loss += loss.item()

            all_probs = np.concatenate(all_probs, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            all_attrs = np.concatenate(all_attrs, axis=0)
            eval_avg_loss /= len(val_dataloader)

            logger.log(f'===> epoch[{epoch:03d}/{args.num_epochs:03d}], training loss: {avg_loss:.4f}, eval loss: {eval_avg_loss:.4f}')

            eval_overall_acc, eval_esaccs_by_attrs, eval_overall_auc, eval_esaucs_by_attrs, eval_aucs_by_attrs, eval_dpds, eval_eods, eval_between_group_disparity,eval_specificity,eval_sensitivity,eval_f1,eval_precision = evalute_comprehensive_perf(all_probs, all_labels, all_attrs.T)

            if best_auc <= eval_overall_auc:
                best_auc = eval_overall_auc
                best_acc = eval_overall_acc
                best_ep = epoch
                best_auc_groups = eval_aucs_by_attrs
                best_dpd_groups = eval_dpds
                best_eod_groups = eval_eods
                best_es_acc = eval_esaccs_by_attrs
                best_es_auc = eval_esaucs_by_attrs
                best_between_group_disparity = eval_between_group_disparity

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': eval_avg_loss,
                    'best_acc': best_acc,
                    'best_auc': best_auc,
                    'best_ep': best_ep,
                    'best_auc_groups': best_auc_groups,
                    'best_dpd_groups': best_dpd_groups,
                    'best_eod_groups': best_eod_groups,
                    'best_es_acc': best_es_acc,
                    'best_es_auc': best_es_auc,
                    'best_between_group_disparity': best_between_group_disparity
                }, os.path.join(args.result_dir, f"best.pth"))
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': eval_avg_loss,
                    'best_acc': best_acc,
                    'best_auc': best_auc,
                    'best_ep': best_ep,
                    'best_auc_groups': best_auc_groups,
                    'best_dpd_groups': best_dpd_groups,
                    'best_eod_groups': best_eod_groups,
                    'best_es_acc': best_es_acc,
                    'best_es_auc': best_es_auc,
                    'best_between_group_disparity': best_between_group_disparity
                }, os.path.join(args.result_dir, f"last.pth"))


            if args.result_dir is not None:
                np.savez(os.path.join(args.result_dir, f'pred_gt_ep{epoch:03d}.npz'),
                            val_pred=all_probs, val_gt=all_labels, val_attr=all_attrs)

            logger.log(f'---- best AUC {best_auc:.4f} at epoch {best_ep}')
            logger.log(f'---- best AUC by groups and attributes at epoch {best_ep}')

            logger.logkv('epoch', epoch)
            logger.logkv('trn_loss', round(avg_loss,4))

            logger.logkv('eval_loss', round(eval_avg_loss,4))
            logger.logkv('eval_acc', round(eval_overall_acc,4))
            logger.logkv('eval_auc', round(eval_overall_auc,4))

            for ii in range(len(eval_esaccs_by_attrs)):
                logger.logkv(f'eval_es_acc_attr{ii}', round(eval_esaccs_by_attrs[ii],4))
            for ii in range(len(eval_esaucs_by_attrs)):
                logger.logkv(f'eval_es_auc_attr{ii}', round(eval_esaucs_by_attrs[ii],4))
            for ii in range(len(eval_aucs_by_attrs)):
                for iii in range(len(eval_aucs_by_attrs[ii])):
                    logger.logkv(f'eval_auc_attr{ii}_group{iii}', round(eval_aucs_by_attrs[ii][iii],4))

            for ii in range(len(eval_between_group_disparity)):
                logger.logkv(f'eval_auc_attr{ii}_std_group_disparity', round(eval_between_group_disparity[ii][0],4))
                logger.logkv(f'eval_auc_attr{ii}_max_group_disparity', round(eval_between_group_disparity[ii][1],4))

            for ii in range(len(eval_dpds)):
                logger.logkv(f'eval_dpd_attr{ii}', round(eval_dpds[ii],4))
            for ii in range(len(eval_eods)):
                logger.logkv(f'eval_eod_attr{ii}', round(eval_eods[ii],4))

            logger.dumpkvs()
    
    if args.perf_file != '':
        if os.path.exists(best_global_perf_file):
            with open(best_global_perf_file, 'a') as f:

                esacc_head_str = ', '.join([f'{x:.4f}' for x in best_es_acc]) + ', '
                esauc_head_str = ', '.join([f'{x:.4f}' for x in best_es_auc]) + ', '

                auc_head_str = ''
                for i in range(len(best_auc_groups)):
                    auc_head_str += ', '.join([f'{x:.4f}' for x in best_auc_groups[i]]) + ', '

                group_disparity_str = ''
                for i in range(len(best_between_group_disparity)):
                    group_disparity_str += ', '.join([f'{x:.4f}' for x in best_between_group_disparity[i]]) + ', '
                
                dpd_head_str = ', '.join([f'{x:.4f}' for x in best_dpd_groups]) + ', '
                eod_head_str = ', '.join([f'{x:.4f}' for x in best_eod_groups]) + ', '

                path_str = f'{args.result_dir}_seed{args.seed}_auc{best_auc:.4f}'
                f.write(f'{best_ep}, {best_acc:.4f}, {esacc_head_str} {best_auc:.4f}, {esauc_head_str} {auc_head_str} {dpd_head_str} {eod_head_str} {group_disparity_str} {path_str}\n')

    if args.use_gen_data:
        output_name = f'results/{args.model_arch}使用生成数据-seed{args.seed}_auc{best_auc:.4f}'
    else:
        output_name = f'results/{args.model_arch}不使用生成数据-seed{args.seed}_auc{best_auc:.4f}'
    print(output_name)
    os.rename(args.result_dir, output_name)