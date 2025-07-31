import datetime
import os
import numpy as np
import random
import argparse
import time
import json
import pandas as pd
from collections import Counter
import re
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import einops
from pytorch_lightning import seed_everything
import sys
sys.path.append('.')
from cldm.ddim_hacked import DDIMSampler
from cycle_resnet_src.modules import *
from cycle_resnet_src import logger
import shutil

from share import *
import sys
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import uuid
import gc
import lpips


from pytorch_fid import fid_score
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


#pip install pytorch-fid
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = False

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


parser = argparse.ArgumentParser(description='FairCLIP Training/Fine-Tuning')

parser.add_argument('--seed', default=-1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--num_epochs', default=30, type=int)
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=6e-5, type=float,
                    metavar='W', help='weight decay (default: 6e-5)',
                    dest='weight_decay')

parser.add_argument('--result_dir', default='/root/cloud/ziheng/output/results', type=str)
parser.add_argument('--dataset_dir', default='/root/data/fairvlmed10k', type=str)
# parser.add_argument('--dataset_dir', default='/H_share/data/fairvlmed10k', type=str)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--eval_set', default='test', type=str, help='options: val | test')
# parser.add_argument('--summarized_note_file', default='/H_share/data/fairvlmed10k/gpt-4_summarized_notes.csv',
#                     type=str)
parser.add_argument('--summarized_note_file', default='/root/data/fairvlmed10k/data_summary_all.csv')
parser.add_argument('--text_source', default='note', type=str, help='options: note | label')
parser.add_argument('--perf_file', default='', type=str)
parser.add_argument('--model_arch', default='efficientnet_b4', type=str, help='options: efficientnet_b0 | efficientnet_b1 | efficientnet_b2 | efficientnet_b3 | efficientnet_b4 | efficientnet_b5 | efficientnet_b6 | efficientnet_b7')
parser.add_argument('--pretrained_weights', default='models/efficientnet_b4_pretrain_0.7375.pth', type=str)
# parser.add_argument('--pretrained_weights', default='models/temp.pth', type=str)
parser.add_argument('--mode', type=str,default='confidence', choices=['confidence', 'uncertainty','both'], help='Select confidence or uncertainty or both')
parser.add_argument('--use_gen_data', default=True, type=bool)


def inference(loader,model,args,device):
    model.eval()



    with torch.no_grad():
        all_uncertainty = []
        for i ,(input, _, label_and_attributes, glaucoma_label, base_names) in enumerate(tqdm(loader,desc="uncertainty")):

            T=8
            stride = input.shape[0]
            preds = torch.zeros([stride * T, 1]).to(device)
            for j in range(T):
                unc_input = input + torch.clamp(torch.randn_like(input) * 0.1, -0.3, 0.3)
                unc_input = unc_input.to(device)
                unc_output = model(unc_input)
                preds[j * stride: (j + 1) * stride] = unc_output
            preds = preds.squeeze()
            preds = torch.sigmoid(preds)
            preds = preds.reshape(T,stride, 1)
            preds_mean = torch.mean(preds,dim = 0)
            uncertainty =  -1.0 * torch.sum(preds_mean * torch.log(preds_mean + 1e-7), dim=1, keepdim=True)
            for basename,uncertain,label in zip(base_names,uncertainty,glaucoma_label):
                all_uncertainty.append([basename,uncertain.item(),label.item()])

        return all_uncertainty


def train_resnet(args,global_epoch,metrics):


    set_random_seed(args.seed)
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

            group_disparity_head_str += ', '.join([f'std_group_disparity_attr{x}, max_group_disparity_attr{x}' for x in
                                                   range(len(groups_in_attrs))]) + ', '

            with open(best_global_perf_file, 'w') as f:
                f.write(
                    f'epoch, acc, {esacc_head_str} auc, {esauc_head_str} {auc_head_str} {dpd_head_str} {eod_head_str} {group_disparity_head_str} path\n')

    device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
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
        model = efficientnet_archs[args.model_arch](weights=None)
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
    # if global_epoch == 0:
    #     checkpoint = torch.load(args.pretrained_weights)
    #
    #
    # else:
    #     ckpt_path = os.path.join(args.result_dir,"last.pth")
    #     checkpoint = torch.load(ckpt_path)
    checkpoint = torch.load(args.pretrained_weights,weights_only=False)

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

    train_dataset = fair_vl_med_dataset(args,args.dataset_dir, transform_train, use_gen_data=args.use_gen_data,
                                        subset='training', text_source=args.text_source,
                                        summarized_note_file=args.summarized_note_file, files=train_files)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True, drop_last=True)

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
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.1, 0.1), eps=1e-6, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)



    best_auc = checkpoint['best_auc']
    best_acc = checkpoint['best_acc']
    best_ep = checkpoint['best_ep']
    best_auc_groups = checkpoint['best_auc_groups']
    best_dpd_groups = checkpoint['best_dpd_groups']
    best_eod_groups = checkpoint['best_eod_groups']
    best_es_acc = checkpoint['best_es_acc']
    best_es_auc = checkpoint['best_es_auc']
    best_between_group_disparity = checkpoint['best_between_group_disparity']
    if 'best_specificity' in checkpoint:
        best_specificity= checkpoint['best_specificity']
    else:
        best_specificity = -1
    if 'best_sensitivity' in checkpoint:
        best_sensitivity= checkpoint['best_sensitivity']
    else:
        best_sensitivity = -1
    if 'best_f1' in checkpoint:
        best_f1= checkpoint['best_f1']
    else:
        best_f1 = -1
    if 'best_precision' in checkpoint:
        best_precision= checkpoint['best_precision']
    else:
        best_precision = -1


    start_epoch = 0
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 初始化
    # model_ema = EMA(model, 0.999)
    # model_ema.register()

    confidence_dict = {}
    # 初始化
    ema = EMA_confidence(confidence_dict, 0.999)
    if global_epoch ==0:
        epochs = 1
    else:
        epochs = args.num_epochs

    for epoch in range(epochs):
        avg_loss = 0
        model.train()
        for i,batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            images, _, label_and_attributes, glaucoma_label, base_names = batch
            images = images.to(device)
            # texts = texts.to(device)
            glaucoma_label = glaucoma_label.to(device)
            logits_per_image = model(images)
            logits_per_image = logits_per_image.squeeze()
            sigmoid_logits_per_image = logits_per_image.sigmoid().cpu().detach().numpy()
            glaucoma_label_numpy = glaucoma_label.cpu().detach().numpy()
            for logits, label, basename in zip(sigmoid_logits_per_image, glaucoma_label_numpy, base_names):
                confidence_dict[basename] = [logits, label]
                ema.register(basename,[logits, label])
            loss = bce(logits_per_image, glaucoma_label.float())
            if global_epoch == 0:
                pass
            else:
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
                optimizer.step()
                # model_ema.update()

            avg_loss += loss.item()

            # if i > 5:
            #     break

        avg_loss /= len(train_dataloader)
        ema.update()

        # iterate over val dataset
        model.eval()
        if epoch == epochs - 1:
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
                if  os.path.exists('best.pth'):
                    model.load_state_dict(
                        torch.load(os.path.join(args.result_dir, f"best.pth"))['model_state_dict'])
                else:
                    pass
            else:
                # model_ema.apply_shadow()
                pass




            for batch in loader:
                images, _, label_and_attributes, glaucoma_label, base_name = batch

                images = images.to(device)
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

            logger.log(
                f'===> epoch[{epoch:03d}/{epochs:03d}], training loss: {avg_loss:.4f}, eval loss: {eval_avg_loss:.4f}')

            overall_acc, eval_es_acc, overall_auc, eval_es_auc, eval_aucs_by_attrs, eval_dpds, eval_eods, between_group_disparity,specificity,sensitivity,f1,precision = evalute_comprehensive_perf(
                all_probs, all_labels, all_attrs.T)

            if best_specificity == -1:
                best_specificity = specificity
            if best_sensitivity == -1:
                best_sensitivity = sensitivity
            if best_f1 == -1:
                best_f1 = f1
            if best_precision == -1:
                best_precision = precision

            if best_auc <= overall_auc:
                best_auc = overall_auc
                best_acc = overall_acc
                best_ep = epoch
                best_auc_groups = eval_aucs_by_attrs
                best_dpd_groups = eval_dpds
                best_eod_groups = eval_eods
                best_es_acc = eval_es_acc
                best_es_auc = eval_es_auc
                best_between_group_disparity = between_group_disparity
                best_specificity = specificity
                best_sensitivity= sensitivity
                best_f1 = f1
                best_precision = precision

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': eval_avg_loss,
                    'best_acc':best_acc,
                    'best_auc':best_auc,
                    'best_ep':best_ep,
                    'best_auc_groups':best_auc_groups,
                    'best_dpd_groups':best_dpd_groups,
                    'best_eod_groups':best_eod_groups,
                    'best_es_acc':best_es_acc,
                    'best_es_auc':best_es_auc,
                    'best_between_group_disparity':best_between_group_disparity,
                    'best_specificity' : best_specificity,
                    'best_sensitivity' : best_sensitivity,
                    'best_f1' : best_f1,
                    'best_precision' : best_precision
                }, os.path.join(args.result_dir, f"best.pth"))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': eval_avg_loss,
                'best_acc':best_acc,
                'best_auc':best_auc,
                'best_ep':best_ep,
                'best_auc_groups':best_auc_groups,
                'best_dpd_groups':best_dpd_groups,
                'best_eod_groups':best_eod_groups,
                'best_es_acc':best_es_acc,
                'best_es_auc':best_es_auc,
                'best_between_group_disparity':best_between_group_disparity,
                'best_specificity': best_specificity,
                'best_sensitivity': best_sensitivity,
                'best_f1': best_f1,
                'best_precision': best_precision
            }, os.path.join(args.result_dir, f"last.pth"))


            if loader == val_dataloader:
                # model_ema.restore()
                metrics.update(
                    {'val-best_acc': best_acc, 'val-best_auc': best_auc, 'val-best_specificity': best_specificity,
                     'val-best_sensitivity': best_sensitivity, 'val-best_f1': best_f1, 'val-best_precision': best_precision
                     })
            elif loader == test_dataloader:
                metrics.update(
                    {'test-best_acc': overall_acc, 'test-best_auc': overall_auc, 'test-best_specificity': specificity,
                     'test-best_sensitivity': sensitivity, 'test-best_f1': f1, 'test-best_precision': precision
                     })

            # if args.result_dir is not None:
            #     np.savez(os.path.join(args.result_dir, f'pred_gt_ep{epoch:03d}.npz'),
            #              val_pred=all_probs, val_gt=all_labels, val_attr=all_attrs)

            logger.log(f'---- best AUC {best_auc:.4f} best ACC {best_acc:.4f} best specificity {best_specificity:.4f}\n'
                       f' best sensitivity {best_sensitivity:.4f} best f1 {best_f1:.4f} best precision {best_precision:.4f} at epoch {best_ep}'
                       )
            logger.log(f'---- best AUC by groups and attributes at epoch {best_ep}')

            logger.logkv('epoch', epoch)
            logger.logkv('trn_loss', round(avg_loss, 4))

            logger.logkv('eval_loss', round(eval_avg_loss, 4))
            logger.logkv('eval_acc', round(overall_acc, 4))
            logger.logkv('eval_auc', round(overall_auc, 4))
            logger.logkv('eval_specificity', round(specificity, 4))
            logger.logkv('eval_sensitivity', round(sensitivity, 4))
            logger.logkv('eval_f1', round(f1, 4))
            logger.logkv('eval_precision', round(precision, 4))

            for ii in range(len(eval_es_acc)):
                logger.logkv(f'eval_es_acc_attr{ii}', round(eval_es_acc[ii], 4))
            for ii in range(len(eval_es_auc)):
                logger.logkv(f'eval_es_auc_attr{ii}', round(eval_es_auc[ii], 4))
            for ii in range(len(eval_aucs_by_attrs)):
                for iii in range(len(eval_aucs_by_attrs[ii])):
                    logger.logkv(f'eval_auc_attr{ii}_group{iii}', round(eval_aucs_by_attrs[ii][iii], 4))

            for ii in range(len(between_group_disparity)):
                logger.logkv(f'eval_auc_attr{ii}_std_group_disparity', round(between_group_disparity[ii][0], 4))
                logger.logkv(f'eval_auc_attr{ii}_max_group_disparity', round(between_group_disparity[ii][1], 4))

            for ii in range(len(eval_dpds)):
                logger.logkv(f'eval_dpd_attr{ii}', round(eval_dpds[ii], 4))
            for ii in range(len(eval_eods)):
                logger.logkv(f'eval_eod_attr{ii}', round(eval_eods[ii], 4))

            logger.dumpkvs()

    ema.apply_shadow()
    ema.restore()
    confidence_dict = ema.confidence
    top_n = 0.05 #获取前n%索引
    remove_n = 0.025 #删除前n%索引
    logger.log(f'---- top_n {top_n} remove_n {remove_n}')
    if args.mode == "confidence":
        #置信度模式

        # 选取符合的数据微调controlnet
        all_confidence = np.array(
            [[basename, probability, label] for basename, (probability, label) in confidence_dict.items()])
        gen_file = np.array([i for i in all_confidence if "generate" in i[0]])
        source_file = np.array([i for i in all_confidence if "generate" not in  i[0]])
        gen_t = [abs(float(probability) - float(label)) for basename, probability, label in gen_file]
        source_t = [abs(float(probability) - float(label)) for basename, probability, label in source_file]
        all_t = [abs(float(probability) - float(label)) for basename, probability, label in all_confidence]
        mean_absolute_percentage_error = np.mean(all_t)
        logger.log(f'mean_absolute_percentage_error(越小越好) {mean_absolute_percentage_error}')
        # 源数据集从大到小返回索引
        t_indices = np.argsort(source_t)[::-1]
        # 获取前10%的误判程度高的索引
        topn_t_indices = t_indices[:int(len(t_indices) * top_n)]
        topn_source_path = source_file[topn_t_indices][:, 0]
    elif args.mode == "uncertainty":
        #不确定性模式
        all_uncertainty = inference(train_dataloader, model, args, device)
        gen_file = np.array([i for i in all_uncertainty if "generate" in i[0]])
        source_file = np.array([i for i in all_uncertainty if "generate" not in i[0]])
        gen_t = [float(uncertainty) for basename, uncertainty, label in gen_file]
        source_t = [float(uncertainty) for basename, uncertainty, label in source_file]
        all_t = [float(uncertainty) for basename, uncertainty, label in all_uncertainty]
        mean_uncertainty = np.mean(all_t)
        logger.log(f'mean_uncertainty(越小越好) {mean_uncertainty}')
        # 源数据集从大到小返回索引
        t_indices = np.argsort(source_t)[::-1]
        # 获取前m%的不确定性程度高的索引
        topn_t_indices = t_indices[:int(len(t_indices) * top_n)]
        topn_source_path = source_file[topn_t_indices][:, 0]
    elif args.mode == "both":
        #both = (1-uncertainty)*confidence_error

        all_confidence = np.array(
            [[basename, probability, label] for basename, (probability, label) in confidence_dict.items()])
        all_confidence_df = pd.DataFrame(all_confidence)
        all_confidence_df.columns = ['name','probability','label']
        all_uncertainty = inference(train_dataloader, model, args, device)
        all_uncertainty_df = pd.DataFrame(all_uncertainty)
        all_uncertainty_df.columns = ['name','uncertainty','label']
        merge = pd.merge(left=all_confidence_df,right=all_uncertainty_df,how='inner',on='name')
        merge['probability'] = merge['probability'].astype(float)
        merge['label_x'] = merge['label_x'].astype(float)
        merge['confidence_error'] = abs(merge['probability'] - merge['label_x'])
        merge['both'] = merge['confidence_error'] * (1-merge['uncertainty'])
        merge = merge.sort_values(by='both', ascending=False)

        gen_file = merge[merge['name'].str.contains('generate')][['name','both','label_x']].values
        source_file = merge[~merge['name'].str.contains('generate')][['name','both','label_x']].values
        all_both = merge[['name','both','label_x']].values
        gen_t = [float(both) for basename, both, label in gen_file]
        source_t = [float(both) for basename, both, label in source_file]
        all_t = [float(both) for basename, both, label in all_both]
        mean_both = np.mean(all_t)
        logger.log(f'mean_both(越小越好) {mean_both}')
        # 源数据集从大到小返回索引
        t_indices = np.argsort(source_t)[::-1]
        # 获取前m%的不确定性程度高的索引

        topn_t_indices = t_indices[:int(len(t_indices) * top_n)]
        topn_source_path = source_file[topn_t_indices][:, 0]

    else:
        raise "模式错误"

    # 生成数据从大到小返回索引
    gen_t_indices = np.argsort(gen_t)[::-1]
    if len(gen_t_indices) == 0:
        pass
    else:
        # 获取前50%的误判程度高的索引(生成500张，删除250张)
        topn_gen_t_indices = gen_t_indices[:int(len(t_indices) * remove_n)]
        topn_gen_path = gen_file[topn_gen_t_indices][:, 0]
        # 执行删除
        for path in topn_gen_path:
            path = re.sub('_generate.*',"",path)
            li = glob(os.path.join(path, "*"))
            for remove_i in li:
                os.remove(remove_i)

    output_path = topn_source_path

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
                f.write(
                    f'{best_ep}, {best_acc:.4f}, {esacc_head_str} {best_auc:.4f}, {esauc_head_str} {auc_head_str} {dpd_head_str} {eod_head_str} {group_disparity_str} {path_str}\n')
    return output_path



def get_last_lightning_cpkpt(result_dir,delete_old=True):
    files_names = glob(os.path.join(result_dir,'lightning_logs/*'))
    latest_file = sorted(files_names, key=lambda x: x.split("_")[-1], reverse=True)
    for file in latest_file:
        ckpt_path = os.path.join(file, "checkpoints", "*")
        ckpt_path = glob(ckpt_path)
        if len(ckpt_path) == 0:
            continue
        else:
            ckpt_path = ckpt_path[0]
            if delete_old:
                latest_file.remove(file)
                for i in latest_file:
                    shutil.rmtree(i)

            return ckpt_path
    raise FileNotFoundError("No checkpoints found")
def get_last_global(result_dir):
    files_names = glob(os.path.join(result_dir,'log_train*'))
    sort_files_names = sorted(files_names, key=lambda x: x.split("global")[-1].split(".txt")[0], reverse=True)
    last = sort_files_names[0]
    last_global = int(last.split("global")[-1].split(".txt")[0])
    return last_global

def train_control(args,global_epoch,topn_file,created_model):
    # Configs
    resume_path = './models/control_sd21_ini.ckpt'
    batch_size = 3
    logger_freq = 300
    learning_rate = 1e-5
    sd_locked = False
    only_mid_control = False
    finetune_epoch = 1

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = created_model
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    if global_epoch == 0:
        ckpt_path = ""
        now_epoch = 0
    else:
        ckpt_path = get_last_lightning_cpkpt(args.result_dir)
        now_epoch = int(re.findall('epoch=(\d+)',ckpt_path)[0])
    max_epoch = now_epoch + finetune_epoch+1



    # Misc
    dataset = MyDataset(args=args ,file_list=topn_file)
    dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq, log_images_kwargs={"ddim_steps": 25},disabled=True)
    save_checkpoint = pl.callbacks.ModelCheckpoint(every_n_epochs=1, save_top_k=1)
    trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, save_checkpoint], max_epochs=max_epoch,weights_save_path=args.result_dir)

    # Train!
    trainer.fit(model, dataloader,ckpt_path=ckpt_path)

    del trainer,model,dataloader,logger,dataset
def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta,model,ddim_sampler):
    with torch.no_grad():
        # input_image = HWC3(input_image)
        # detected_map = apply_uniformer(resize_image(input_image, detect_resolution))
        # img = resize_image(input_image, image_resolution)
        input_image = input_image.squeeze()
        H, W, C = input_image.shape

        # detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        # control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = input_image
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)


        # print(prompt, a_prompt)
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt[0] + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        # x_samples = (x_samples * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [input_image.permute(2, 0, 1).cpu()] + results

def gen_new_image(args,topn_file,created_model,m2,s2,global_epoch,metrics):
    model = created_model
    ckpt_path = get_last_lightning_cpkpt(args.result_dir)
    model.load_state_dict(
        load_state_dict(ckpt_path, location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    dataset = MyDataset(args=args,gen_data=True,file_list=topn_file)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=False)

    save_dir = os.path.join(args.result_dir,f"sd_gen{global_epoch}")
    os.makedirs(save_dir, exist_ok=True)
    generate_infos = {}
    for i,item in enumerate(tqdm(dataloader)):

        target = item['jpg'].to('cuda')
        prompt = item['txt']
        source = item['hint'].to('cuda')
        race = item['race'].item()
        glaucoma_label = item['glaucoma_label'].item()
        gender=item['gender'].item()
        ethnicity=item['ethnicity'].item()
        language=item['language'].item()
        output_name = item['name'][0]
        # rs_target = item['rs_target']
        output = process(source, prompt, '', '', 1, 512, 512, ddim_steps=30, guess_mode=False, strength=1.0, scale=9.0, seed=np.random.randint(0, 65536, 1), eta=0,model=model,ddim_sampler=ddim_sampler)
        generated_uuid = str(uuid.uuid4())[:4]
        for j in range(0, len(output)):

            img = output[j]
            img = transforms.ToPILImage()(img)
            if j == 0:
                print(f'{generated_uuid}')
                img.save(os.path.join(save_dir, f'{output_name}_{generated_uuid}_mask.png'))
                with open(os.path.join(save_dir, f'{output_name}_{generated_uuid}_prompt.txt'), 'w') as f:
                    f.write(prompt[0])
            else:
                png_path = os.path.join(save_dir, f'{output_name}_{generated_uuid}_generate.png')
                img.save(png_path)
                generate_infos[png_path] = {"race":race, "glaucoma_label":glaucoma_label, "gender":gender, "ethnicity":ethnicity, "language":language}
    generate = pd.DataFrame(generate_infos).T.reset_index(names='path')
    generate = generate.reset_index()
    def fid_apply_process(x):
        path = x['path'].values
        if len(path)==1:
            path = np.tile(path, 2)
        nonlocal m2,s2
        fid, m2, s2 = fid_score.calculate_fid_given_paths([path, glob(
            os.path.join(args.dataset_dir, "FairCLIP调色后清洗/training_slo_fundus/*"))], 16, 'cuda', 2048, m2=m2,
                                                          s2=s2)
        return fid

    def ssim_apply_process(x):
        path = x['path'].values
        # 计算SSIM和MS-SSIM分数
        # X: 生成图片路径列表，Y: 真实图片路径列表
        if len(path) == 1:
            path = np.tile(path, 2)

        real_paths = []
        for p in path:
            basename = os.path.basename(p)
            real_file_name = re.findall(r'data_\d+',basename)[0]
            real_file_name = real_file_name+".png"
            real_paths.append(os.path.join(args.dataset_dir, "FairCLIP调色后清洗/training_slo_fundus",real_file_name))

        # 读取图片并转为tensor
        imgs_gen = [transforms.ToTensor()(transforms.Resize((256, 256))(transforms.ToPILImage()(np.array(Image.open(p).convert('RGB'))))) for p in path]
        imgs_real = [transforms.ToTensor()(transforms.Resize((256, 256))(transforms.ToPILImage()(np.array(Image.open(p).convert('RGB'))))) for p in real_paths]
        X = torch.stack(imgs_gen, dim=0) * 255
        Y = torch.stack(imgs_real, dim=0) * 255
        X = X.type(torch.uint8).float().cuda()
        Y = Y.type(torch.uint8).float().cuda()
        # 计算ssim和ms-ssim
        ssim_val = ssim(X, Y, data_range=255, size_average=True)
        ms_ssim_val = ms_ssim(X, Y, data_range=255, size_average=True)
        return {'ssim': float(ssim_val.cpu().numpy()), 'ms_ssim': float(ms_ssim_val.cpu().numpy())}

    def lpips_apply_process(x):
        # 读取图片路径
        path = x['path'].values
        if len(path) == 1:
            path = np.tile(path, 2)
        # 获取真实图片路径
        real_paths = []
        for p in path:
            basename = os.path.basename(p)
            real_file_name = re.findall(r'data_\d+',basename)[0]
            real_file_name = real_file_name+".png"
            real_paths.append(os.path.join(args.dataset_dir, "FairCLIP调色后清洗/training_slo_fundus",real_file_name))

        # 读取图片并转为tensor，归一化到[-1,1]
        imgs_gen = [transforms.ToTensor()(transforms.Resize((256, 256))(transforms.ToPILImage()(np.array(Image.open(p).convert('RGB'))))) for p in path]
        imgs_real = [transforms.ToTensor()(transforms.Resize((256, 256))(transforms.ToPILImage()(np.array(Image.open(p).convert('RGB'))))) for p in real_paths]
        X = torch.stack(imgs_gen, dim=0)
        Y = torch.stack(imgs_real, dim=0)
        # 归一化到[-1,1]
        X = (X * 2 - 1).cuda()
        Y = (Y * 2 - 1).cuda()
        # 加载lpips模型
        loss_fn_alex = lpips.LPIPS(net='alex').cuda()
        # 计算LPIPS分数
        # 逐对计算平均
        alex_scores = []
        for i in range(X.shape[0]):
            img0 = X[i].unsqueeze(0)
            img1 = Y[i].unsqueeze(0)
            d_alex = loss_fn_alex(img0, img1)
            alex_scores.append(float(d_alex.cpu().detach().numpy()))
        return {'lpips_alex': float(np.mean(alex_scores))}
    

    race = generate.groupby('race').apply(fid_apply_process)
    gender = generate.groupby('gender').apply(fid_apply_process)
    ethnicity = generate.groupby('ethnicity').apply(fid_apply_process)
    language = generate.groupby('language').apply(fid_apply_process)
    glaucoma_label = generate.groupby('glaucoma_label').apply(fid_apply_process)
    all_fid, m2, s2 = fid_score.calculate_fid_given_paths([glob(save_dir+"/*.png"), glob(
        os.path.join(args.dataset_dir, "FairCLIP调色后清洗/training_slo_fundus/*"))], 16, 'cuda', 2048, m2=m2,s2=s2)

    race_ssim = generate.groupby('race').apply(ssim_apply_process)
    gender_ssim = generate.groupby('gender').apply(ssim_apply_process)
    ethnicity_ssim = generate.groupby('ethnicity').apply(ssim_apply_process)
    language_ssim = generate.groupby('language').apply(ssim_apply_process)
    glaucoma_label_ssim = generate.groupby('glaucoma_label').apply(ssim_apply_process)
    

    race_lpips = generate.groupby('race').apply(lpips_apply_process)
    gender_lpips = generate.groupby('gender').apply(lpips_apply_process)
    ethnicity_lpips = generate.groupby('ethnicity').apply(lpips_apply_process)
    language_lpips = generate.groupby('language').apply(lpips_apply_process)
    glaucoma_label_lpips = generate.groupby('glaucoma_label').apply(lpips_apply_process)
    
    # 计算所有生成图片的ssim并求平均
    all_gen_paths = generate['path'].values
    all_real_paths = []
    for p in all_gen_paths:
        basename = os.path.basename(p)
        real_file_name = re.findall(r'data_\d+', basename)[0]
        real_file_name = real_file_name + ".png"
        all_real_paths.append(os.path.join(args.dataset_dir, "FairCLIP调色后清洗/training_slo_fundus", real_file_name))

    # 读取图片并转为tensor，归一化到[-1,1]
    imgs_gen = [transforms.ToTensor()(transforms.Resize((256, 256))(transforms.ToPILImage()(np.array(Image.open(p).convert('RGB'))))) for p in all_gen_paths]
    imgs_real = [transforms.ToTensor()(transforms.Resize((256, 256))(transforms.ToPILImage()(np.array(Image.open(p).convert('RGB'))))) for p in all_real_paths]
    X = torch.stack(imgs_gen, dim=0)
    Y = torch.stack(imgs_real, dim=0)
    # 计算SSIM分数
    ssim_scores = []
    ms_ssim_scores = []
    for i in range(X.shape[0]):
        img_gen = X[i].unsqueeze(0)
        img_real = Y[i].unsqueeze(0)
        score = ssim(img_gen, img_real, data_range=2, size_average=True)
        ssim_scores.append(float(score.cpu().detach().numpy()))
        ms_ssim_score = ms_ssim(img_gen, img_real, data_range=2, size_average=True)
        ms_ssim_scores.append(float(ms_ssim_score.cpu().detach().numpy()))

    all_ssim = {'all_ssim': float(np.mean(ssim_scores)),'all_ms_ssim': float(np.mean(ms_ssim_scores)),"race_ssim":race_ssim.to_dict(),"gender_ssim":gender_ssim.to_dict(),"ethnicity_ssim":ethnicity_ssim.to_dict(),"language_ssim":language_ssim.to_dict(),"glaucoma_label_ssim":glaucoma_label_ssim.to_dict()}

    # 计算所有生成图片的lpips并求平均
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    lpips_scores = []
    X = (X * 2 - 1).cuda()
    Y = (Y * 2 - 1).cuda()
    for i in range(X.shape[0]):
        img_gen = X[i].unsqueeze(0)
        img_real = Y[i].unsqueeze(0)
        d_alex = loss_fn_alex(img_gen, img_real)    
        lpips_scores.append(float(d_alex.cpu().detach().numpy()))
    all_lpips = {'all_lpips_alex': float(np.mean(lpips_scores)),"race_lpips":race_lpips.to_dict(),"gender_lpips":gender_lpips.to_dict(),"ethnicity_lpips":ethnicity_lpips.to_dict(),"language_lpips":language_lpips.to_dict(),"glaucoma_label_lpips":glaucoma_label_lpips.to_dict()}



    
    with open(os.path.join(args.result_dir,f'ssim.jsonl'), 'a', encoding='utf-8') as outfile:
        json.dump(all_ssim, outfile, ensure_ascii=False)
        outfile.write('\n')
    with open(os.path.join(args.result_dir,f'lpips.jsonl'), 'a', encoding='utf-8') as outfile:
        json.dump(all_lpips, outfile, ensure_ascii=False)
        outfile.write('\n')




                                                      
    
    fid = {'global_epoch':global_epoch,"all_fid":all_fid,'glaucoma_label':glaucoma_label.to_dict(),'race':race.to_dict(),'gender':gender.to_dict(),'ethnicity':ethnicity.to_dict(),'language':language.to_dict()}


    with open(os.path.join(args.result_dir,f'fid.jsonl'), 'a', encoding='utf-8') as outfile:
        json.dump(fid, outfile, ensure_ascii=False)
        outfile.write('\n')
    metrics['all_fid'] = fid['all_fid']
    for k, v in fid.items():
        if k == 'all_fid' or k == "global_epoch":
            continue
        else:
            for f_k, f_v in v.items():
                metrics[f'fid-{k}-{f_k}'] = f_v
    
    logger.log(f'fid: {fid}')


    del model,dataset,dataloader,ddim_sampler
    return m2,s2



if __name__ == '__main__':
    args = parser.parse_args()
    # 获取当前日期
    current_date = datetime.datetime.now()
    # 格式化日期为字符串，例如：2024-09-25
    date_string = current_date.strftime('%m-%d')
    resume = False
    if resume:
        args.seed = 87620
        args.result_dir = f'/root/cloud/ziheng/cycle-{args.mode}-{args.model_arch}-seed{args.seed}_{args.model_arch}永远加载预训练模型'
        if not os.path.exists(args.result_dir):
            global_epoch = 0
        else:
            global_epoch = get_last_global(args.result_dir)
    else:
        global_epoch = 0
        if args.seed < 0:
            args.seed = int(np.random.randint(100000, size=1)[0])
        args.result_dir = f'/root/cloud/ziheng/{date_string}-cycle-{args.mode}-{args.model_arch}-seed{args.seed}_{args.model_arch}永远加载预训练模型'
        logger.log(f'===> random seed: {args.seed}')
        logger.configure(dir=args.result_dir, log_suffix=f'train-global{global_epoch}')
        with open(os.path.join(args.result_dir, f'args_train.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    created_model = create_model('./models/cldm_v21.yaml').cpu()
    wandb.init(dir=os.path.join(args.result_dir,'wandb'), project="cycle_train",
               name=args.result_dir, config=args, job_type=f'train', entity="ziheng-wang",mode='offline')
    m2, s2 = None, None
    while True:
        metrics = {}
        logger.configure(dir=args.result_dir, log_suffix=f'train-global{global_epoch}')
        logger.log(f'global epoch: {global_epoch}')
        topn_file = train_resnet(args,global_epoch=global_epoch,metrics=metrics)
        torch.cuda.empty_cache()
        # topn_file = topn_file[:3]
        topn_file = list(topn_file)
        train_control(args=args,global_epoch= global_epoch,topn_file= topn_file,created_model=created_model)
        torch.cuda.empty_cache()
        gc.collect()

        m2,s2 = gen_new_image(args=args,topn_file=topn_file,created_model=created_model,m2=m2,s2=s2,global_epoch=global_epoch,metrics=metrics)
        wandb.log(metrics, step=global_epoch, commit=True)

        torch.cuda.empty_cache()
        global_epoch += 1
