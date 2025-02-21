import os
import pandas as pd
from PIL import Image
import json
# import cv2
import numpy as np

from torch.utils.data import Dataset

class eyeDataset(Dataset):
    def __init__(self, data_path, split='train', prompt_type='basic', source_transform=None, target_transform=None):
        self.path = data_path
        self.split = split
        assert split in ['train', 'val', 'test']
        assert prompt_type in ['basic', 'summary', 'full']
        self.prompt_type = prompt_type
        self.source_transform = source_transform
        self.target_transform = target_transform
        summary = os.path.join(self.path, 'data_summary_new.csv')
        self.summary_data = pd.read_csv(summary)

        self.source_path = os.path.join(self.path, 'mask')
        self.target_path = os.path.join(self.path, 'imgdata')

        if self.split == 'train':
            self.data = self.summary_data[self.summary_data['use'] == 'training']
            self.target_path = os.path.join(self.target_path, 'training_slo_fundus')
        elif self.split == 'val':
            self.data = self.summary_data[self.summary_data['use'] == 'validation']
            self.target_path = os.path.join(self.target_path, 'val_slo_fundus')
        elif self.split == 'test':
            self.data = self.summary_data[self.summary_data['use'] == 'test']
            self.target_path = os.path.join(self.target_path, 'test_slo_fundus')

        self.data.reset_index(drop=True, inplace=True)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # prompt = self.data['gpt4_summary'][idx]
        if self.prompt_type == 'basic':
            prompt = 'Age '+ str(self.data.iloc[idx, 1]) + ', ' + ', '.join(self.data.iloc[idx, 2:7].values.tolist())
        elif self.prompt_type == 'summary':
            prompt = self.data['gpt4_summary'][idx]
        elif self.prompt_type == 'full':
            prompt = 'Age '+ str(self.data.iloc[idx, 1]) + ', ' + ', '.join(self.data.iloc[idx, 2:7].values.tolist())
            prompt += ', ' + self.data['gpt4_summary'][idx]
        file_name = self.data['png_filename'][idx]
        source_file = os.path.join(self.source_path, file_name)
        target_file = os.path.join(self.target_path, file_name)

        source = Image.open(source_file).convert("RGB")
        target = Image.open(target_file).convert("RGB")

        source = source.resize((512, 512))
        target = target.resize((512, 512))
        rs_target = np.array(target).astype(np.float32)

        source = np.array(source)
        target = np.array(target)

        # if self.source_transform:
        #     source = self.source_transform(source)
        # if self.target_transform:
        #     target = self.target_transform(target)

        source = source.astype(np.float32) / 255.0

        target = (target.astype(np.float32) / 127.5) - 1.0
        # target = target.permute(1, 2, 0)
        # source = source.permute(1, 2, 0)

        return dict(jpg=target, txt=prompt, hint=source, rs_target=rs_target)