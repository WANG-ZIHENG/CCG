import datetime
import json
import os
import wandb
import numpy as np
import re
from share import *
import sys
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import pandas as pd
import argparse
import torch
import argparse
from glob import glob
import gc
import random
from cycle_resnet_src import logger
from cycle_train import get_last_lightning_cpkpt,gen_new_image,get_last_global
parser = argparse.ArgumentParser(description='FairCLIP Training/Fine-Tuning')
parser.add_argument('--dataset_dir', default='/root/autodl-tmp/fairvlmed10k', type=str)
# parser.add_argument('--dataset_dir', default='/H_share/data/fairvlmed10k', type=str)

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
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    if global_epoch == 0:
        model.load_state_dict(load_state_dict(resume_path, location='cpu'))

        ckpt_path = None
        now_epoch = 0
    else:
        ckpt_path = get_last_lightning_cpkpt(args.result_dir)
        now_epoch = int(re.findall('epoch=(\d+)',ckpt_path)[0])
    max_epoch = now_epoch + finetune_epoch+1



    # Misc
    dataset = MyDataset(args=args ,file_list=topn_file)
    dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq, log_images_kwargs={"ddim_steps": 25},disabled=True)
    save_checkpoint = pl.callbacks.ModelCheckpoint(every_n_epochs=finetune_epoch, save_top_k=1)
    trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, save_checkpoint], max_epochs=max_epoch,weights_save_path=args.result_dir)

    # Train!
    trainer.fit(model, dataloader,ckpt_path=ckpt_path)

    del trainer,model,dataloader,logger,dataset



created_model = create_model('./models/cldm_v21.yaml').cpu()
args = parser.parse_args()

data = glob(os.path.join(args.dataset_dir, 'Training', "*.npz"))
with open(os.path.join(args.dataset_dir,'filter_file.txt'),'r') as f:
    file_list = f.read()
    file_list = file_list.split('\n')
    file_list.remove("")
    file_list = [i.replace('.png','.npz') for i in file_list]
    file_list = set(file_list)
data = [i for i in data if os.path.basename(i) in file_list]

# 获取当前日期
current_date = datetime.datetime.now()
# 格式化日期为字符串，例如：2024-09-25
date_string = current_date.strftime('%m-%d')
resume = False
if resume:
    args.seed = 87620
    # args.result_dir = f'cycle-{args.mode}-resnet18-seed{args.seed}_7.7新指标'
    if not os.path.exists(args.result_dir):
        global_epoch = 0
    else:
        global_epoch = get_last_global(args.result_dir)
else:
    global_epoch = 0

    args.seed = int(np.random.randint(100000, size=1)[0])
    args.result_dir = f'{date_string}-训练controlnet-seed{args.seed}每次30张没病的图片'
    logger.configure(dir=args.result_dir, log_suffix=f'train')

    with open(os.path.join(args.result_dir, f'args_train.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
wandb.init(dir=os.path.join(args.result_dir,'wandb'), project="cycle_train",
           name=args.result_dir, config=args, job_type=f'train', entity="ziheng-wang",mode="offline")
m2,s2=None,None
while True:
    metrics = {}

    #选择正常的图
    summary = os.path.join(args.dataset_dir, 'data_summary.csv')
    summary_data = pd.read_csv(summary)
    filenames = set(summary_data[summary_data['glaucoma'] == 'yes']['filename'])
    li = []
    for i in data:
        filename = os.path.basename(i)
        if filename in filenames:
            li.append(i)
    # li = li[:30]

    train_control(args=args, global_epoch=global_epoch, topn_file=li, created_model=created_model)
    torch.cuda.empty_cache()
    gc.collect()
    li = random.sample(li, 30)
    # li = li[:30]
    fid,m2,s2 = gen_new_image(args=args, topn_file=li, created_model=created_model,global_epoch=global_epoch,m2=m2,s2=s2,metrics=metrics)
    wandb.log(metrics, step=global_epoch, commit=True)
    logger.log(f'{global_epoch} fid: {fid}')
    global_epoch += 1
