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
from transformers import ConvNextV2ForImageClassification, ConvNextV2Config

import sys
sys.path.append('.')

from cycle_resnet_src.modules import *
from cycle_resnet_src import logger


class ModelWrapper(nn.Module):
    """包装器类，统一不同模型的输出格式"""
    def __init__(self, model, is_convnextv2=False):
        super().__init__()
        self.model = model
        self.is_convnextv2 = is_convnextv2
    
    def forward(self, x):
        if self.is_convnextv2:
            # ConvNextV2 返回一个对象，需要提取 logits
            outputs = self.model(x)
            return outputs.logits
        else:
            # EfficientNet 和 ResNet 直接返回 tensor
            return self.model(x)


parser = argparse.ArgumentParser(description='FairCLIP Training/Fine-Tuning')

parser.add_argument('--seed', default=-1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--num_epochs', default=150, type=int,
                    help='total fine-tune epochs for the standalone baseline')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=6e-5, type=float,
                    metavar='W', help='weight decay', dest='weight_decay')

parser.add_argument('--result_dir', default='./results/baseline', type=str)
parser.add_argument('--dataset_dir', default='/root/autodl-tmp/data/fairvlmed10k', type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--workers', default=12, type=int)
parser.add_argument('--eval_set', default='test', type=str, help='options: val | test')
parser.add_argument('--summarized_note_file',
                    default='/root/autodl-tmp/data/fairvlmed10k/data_summary_all.csv',
                    type=str,
                    help='combined with filter_file.txt yields the 7363-row subset')
parser.add_argument('--text_source', default='note', type=str, help='options: note | label')
parser.add_argument('--perf_file', default='', type=str)
parser.add_argument('--model_arch', default='efficientnet_b0', type=str,
                    help='options: efficientnet_b0..b7 | convnextv2_tiny_1k_224 | '
                         'convnextv2_base_1k_224 | convnextv2_large_1k_224 | '
                         'convnextv2_tiny_22k_384 | convnextv2_base_22k_384 | '
                         'convnextv2_large_22k_384')
parser.add_argument('--pretrained_weights', default='', type=str,
                    help='optional .pth path; empty = ImageNet init')

def str_to_bool(v):
    """将字符串转换为布尔值"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser.add_argument('--use_gen_data', default=False, type=str_to_bool)

parser.add_argument('--extra_train_data_dir',
                    default='',
                    type=str,
                    help='Extra directory containing npz files to append to the training split')
parser.add_argument('--exclude_original_label', default=None, type=int,
                    help='Exclude original training samples with this glaucoma label (0 or 1). None means no exclusion.')
parser.add_argument('--exclude_generated_label', default=None, type=int,
                    help='Exclude generated training samples with this glaucoma label (0 or 1). None means no exclusion.')
parser.add_argument('--balance_attribute', default=None, type=str,
                    help='Balance data by specified attribute using generated data. Options: None | gender | race | age ')
parser.add_argument('--cup_disc_threshold', default=0.6, type=float,
                    help='Threshold for cup_disc_ratio to mark generated samples as glaucoma (default: 0.6)')
parser.add_argument('--samples_per_group', default=2, type=int,
                    help='Number of samples to randomly select per group when balancing (default: 2)')

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
    
    # ConvNeXt V2 模型架构列表
    convnextv2_archs = [
        'convnextv2_tiny_1k_224',
        'convnextv2_base_1k_224',
        'convnextv2_large_1k_224',
        'convnextv2_tiny_22k_384',
        'convnextv2_base_22k_384',
        'convnextv2_large_22k_384',
    ]
    
    # ConvNeXt V2 模型架构到 Hugging Face 预训练模型名称的映射
    convnextv2_hf_models = {
        'convnextv2_tiny_1k_224': 'facebook/convnextv2-tiny-1k-224',
        'convnextv2_base_1k_224': 'facebook/convnextv2-base-1k-224',
        'convnextv2_large_1k_224': 'facebook/convnextv2-large-1k-224',
        'convnextv2_tiny_22k_384': 'facebook/convnextv2-tiny-22k-384',
        'convnextv2_base_22k_384': 'facebook/convnextv2-base-22k-384',
        'convnextv2_large_22k_384': 'facebook/convnextv2-large-22k-384',
    }
    
    if args.model_arch in efficientnet_archs:
        base_model = efficientnet_archs[args.model_arch](weights=True)
        num_features = base_model.classifier[1].in_features
        num_classes = 1  # 假设是二分类任务
        base_model.classifier[1] = nn.Linear(num_features, num_classes)
        model = ModelWrapper(base_model, is_convnextv2=False)
    
    elif args.model_arch == 'resnet18':
        base_model = models.resnet18(weights=None)
        num_features = base_model.fc.in_features
        num_classes = 1  # 假设是二分类任务
        base_model.fc = nn.Linear(num_features, num_classes)
        model = ModelWrapper(base_model, is_convnextv2=False)
    
    elif args.model_arch in convnextv2_archs:
        # 使用 ConvNeXt V2 预训练模型
        # 判断是否要加载 .pth 文件
        use_pth_file = args.pretrained_weights and args.pretrained_weights.endswith('.pth') and os.path.exists(args.pretrained_weights)
        
        if use_pth_file:
            # 如果提供了 .pth 文件路径，先从 Hugging Face 加载基础模型结构
            hf_model_name = convnextv2_hf_models.get(args.model_arch)
            if hf_model_name is None:
                raise ValueError(f"未找到模型架构 {args.model_arch} 对应的 Hugging Face 模型名称")
            
            logger.log(f"从 Hugging Face 加载基础模型结构: {hf_model_name}")
            base_model = ConvNextV2ForImageClassification.from_pretrained(
                hf_model_name,
                num_labels=1,  # 二分类任务
                ignore_mismatched_sizes=True  # 允许分类头大小不匹配
            )
            model = ModelWrapper(base_model, is_convnextv2=True)
            logger.log(f"基础模型结构加载完成，稍后将加载 .pth 文件中的权重: {args.pretrained_weights}")
        else:
            # 如果没有提供 .pth 文件，使用 Hugging Face 预训练模型
            pretrained_model_name = convnextv2_hf_models.get(args.model_arch)
            if pretrained_model_name is None:
                raise ValueError(f"未找到模型架构 {args.model_arch} 对应的 Hugging Face 模型名称")
            
            logger.log(f"从 Hugging Face 加载预训练模型: {pretrained_model_name}")
            base_model = ConvNextV2ForImageClassification.from_pretrained(
                pretrained_model_name,
                num_labels=1,  # 二分类任务
                ignore_mismatched_sizes=True  # 允许分类头大小不匹配
            )
            model = ModelWrapper(base_model, is_convnextv2=True)
    
    else:
        raise ValueError(f"暂不支持的模型架构: {args.model_arch}")
    
    model = model.to(device)

    filter_files = 'filter_file.txt'
    test_files = None
    # 数据增强的transforms
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        # transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # 测试集的transforms
    transform_test = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dataset = fair_vl_med_dataset(args,args.dataset_dir, transform_train,use_gen_data=args.use_gen_data, subset='training', text_source=args.text_source, summarized_note_file=args.summarized_note_file,files=filter_files, exclude_original_label=args.exclude_original_label, exclude_generated_label=args.exclude_generated_label, extra_train_data_dir=args.extra_train_data_dir, balance_attribute=args.balance_attribute)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,)

    val_dataset = fair_vl_med_dataset(args,args.dataset_dir, transform_test, summarized_note_file=args.summarized_note_file, subset='validation',files=filter_files)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    test_dataset = fair_vl_med_dataset(args,args.dataset_dir, transform_test, summarized_note_file=args.summarized_note_file, subset='test',files=filter_files)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)



    logger.log(f'# of training samples: {train_dataset.__len__()}, # of val samples: {val_dataset.__len__()}')



    group_size_on_race, group_size_on_gender, group_size_on_ethnicity = count_number_of_groups(train_dataset)
    logger.log(f'group size on race in training set: {group_size_on_race}')
    logger.log(f'group size on gender in training set: {group_size_on_gender}')
    logger.log(f'group size on ethnicity in training set: {group_size_on_ethnicity}')
    group_size_on_race, group_size_on_gender, group_size_on_ethnicity = count_number_of_groups(test_dataset)
    logger.log(f'group size on race in test set: {group_size_on_race}')
    logger.log(f'group size on gender in test set: {group_size_on_gender}')
    logger.log(f'group size on ethnicity in test set: {group_size_on_ethnicity}')
    group_size_on_race, group_size_on_gender, group_size_on_ethnicity = count_number_of_groups(val_dataset)
    logger.log(f'group size on race in val set: {group_size_on_race}')
    logger.log(f'group size on gender in val set: {group_size_on_gender}')
    logger.log(f'group size on ethnicity in val set: {group_size_on_ethnicity}')


    bce = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.1, 0.1), eps=1e-6,weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 判断是否要加载 .pth 文件（仅模型权重，如果之前没有定义则在这里定义）
    if 'use_pth_file' not in locals():
        use_pth_file = args.pretrained_weights and args.pretrained_weights.endswith('.pth') and os.path.exists(args.pretrained_weights)
    
    # 初始化 start_epoch
    start_epoch = 0
    
    if args.pretrained_weights != "":
        checkpoint = torch.load(args.pretrained_weights, weights_only=False, map_location=device)
        
        # 检查 checkpoint 格式

        state_dict = checkpoint['model']
        
        # 加载模型权重（使用 strict=False 允许部分键不匹配）
        try:
            model.load_state_dict(state_dict, strict=False)
            logger.log(f"成功加载模型权重从: {args.pretrained_weights}")
        except Exception as e:
            logger.log(f"加载模型权重时出现警告: {e}")
            logger.log("尝试继续训练...")
        
        # 尝试加载训练状态（如果存在）
        if isinstance(checkpoint, dict):
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                logger.log(f"从 checkpoint 恢复训练，起始 epoch: {start_epoch}")
            else:
                start_epoch = 0
                logger.log("checkpoint 中没有找到 epoch 信息，从 epoch 0 开始训练")
            
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger.log("成功加载 optimizer 状态")
                except Exception as e:
                    logger.log(f"加载 optimizer 状态时出现错误: {e}，将使用新的 optimizer")
            else:
                logger.log("checkpoint 中没有找到 optimizer 状态，将使用新的 optimizer")
        else:
            start_epoch = 0
            logger.log("checkpoint 格式不是字典，从 epoch 0 开始训练")

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

            eval_overall_acc, eval_esaccs_by_attrs, eval_overall_auc, eval_esaucs_by_attrs, eval_aucs_by_attrs, eval_dpds, eval_eods, eval_between_group_disparity, eval_specificity, eval_sensitivity, eval_f1, eval_precision, eval_mcc, eval_qwk = evalute_comprehensive_perf(all_probs, all_labels, all_attrs.T)

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
            logger.logkv('eval_mcc', round(eval_mcc,4))
            logger.logkv('eval_qwk', round(eval_qwk,4))

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

    # 从数据集目录路径中提取数据集名称
    dataset_name = os.path.basename(args.dataset_dir)
    
    if args.use_gen_data:
        # 只有使用生成数据时才会有 balance_attribute
        balance_suffix = ''
        if args.balance_attribute:
            balance_suffix = f'_balance{args.balance_attribute}'
        # 将 cup_disc_threshold 也添加到文件名以便区分不同阈值设置
        cup_suffix = ''
        if getattr(args, 'cup_disc_threshold', None) is not None:
            cup_suffix = f'_cupth{args.cup_disc_threshold}'
        samp_suffix = ''
        if getattr(args, 'samples_per_group', None) is not None:
            samp_suffix = f'_samp{args.samples_per_group}'
        output_name = f'results/{args.model_arch}_{dataset_name}使用生成数据{balance_suffix}{cup_suffix}{samp_suffix}-seed{args.seed}_auc{best_auc:.4f}'
    else:
        output_name = f'results/{args.model_arch}_{dataset_name}不使用生成数据-seed{args.seed}_auc{best_auc:.4f}'
    print(output_name)
    os.rename(args.result_dir, output_name)