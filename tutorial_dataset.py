import json
import cv2
import numpy as np
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, RandomHorizontalFlip, RandomVerticalFlip, Compose,ToPILImage
import torchvision.transforms as F
import os
import pandas as pd
from glob import glob
def random_crop(image,start_h,start_w, crop_height, crop_width):
    if image.shape[0] < crop_height or image.shape[1] < crop_width:
        raise ValueError("Image dimensions should be larger than crop dimensions")
    cropped_image = image[start_h:start_h+crop_height, start_w:start_w+crop_width]

    return cropped_image

class MyDataset(Dataset):
    def __init__(self,args,gen_data = False,file_list = None):
        self.data = []
        
        summary = os.path.join(args.dataset_dir, 'data_summary_all.csv')
        self.summary_data = pd.read_csv(summary).set_index('filename')
        self.gen_data = gen_data
        self.mask_file = os.path.join(args.dataset_dir,'mask')

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.palette = {0: (0, 0, 0),
           -1: (255, 0, 0),  # 红色
           -2: (0, 0, 255)}  # 蓝色

        if file_list != None:
            self.data = file_list
        else:
            self.data = glob(os.path.join(args.dataset_dir, 'Training', "*.npz"))
            with open(os.path.join(args.dataset_dir,'filter_file.txt'),'r') as f:
                file_list = f.read()
                file_list = file_list.split('\n')
                file_list.remove("")
                file_list = [i.replace('.png','.npz') for i in file_list]
                file_list = set(file_list)
            self.data = [i for i in self.data if os.path.basename(i) in file_list]








    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_file = self.data[idx]
        raw_data = np.load(data_file, allow_pickle=True)
        modified_image = raw_data['slo_fundus']
        modified_image = modified_image.astype(np.uint8)
        modified_image = np.array([modified_image, modified_image, modified_image]).transpose(1,2,0)
        clahe_img = np.zeros_like(modified_image)
        for i in range(3):
            clahe_img[:, :, i] = self.clahe.apply(modified_image[:, :, i])
        # mask
        mask_file = os.path.basename(data_file).replace(".npz",".png")
        mask_file = mask_file.split(".")[0]
        mask_file = mask_file + "_predict.png"
        mask_file = os.path.join(self.mask_file,mask_file)
        mask_image = Image.open(mask_file)
        mask_image = mask_image.convert('RGB')
        mask_image = np.array(mask_image)


        source = mask_image #664*789
        target = clahe_img #664*789 mask_image

        index = os.path.basename(data_file)
        info = self.summary_data.loc[index]
        prompt = 'Age ' + str(info['age']) + ', ' + ', '.join(info.iloc[2:7].values.tolist())
        prompt += ', ' + info['gpt4_summary']








        crop_height = crop_width = 512
        start_h = np.random.randint(0, source.shape[0] - crop_height+1)
        start_w = np.random.randint(0, source.shape[1] - crop_width+1)
        source = random_crop(source,start_h,start_w, crop_height, crop_width)
        target = random_crop(target,start_h,start_w, crop_height, crop_width)
        if not self.gen_data:
            r = random.randint(1,100)
            if r >= 50:
                source = cv2.flip(source, 1)
                target = cv2.flip(target, 1)
            r = random.randint(1, 100)
            if r >= 50:
                source = cv2.flip(source, 0)
                target = cv2.flip(target, 0)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        # target = resize(target, (256, 256))
        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0
        name =  os.path.basename(data_file).replace(".npz","")
        glaucoma_label = int(raw_data['glaucoma'].item())
        race = int(raw_data['race'].item())
        gender = int(raw_data['gender'].item())
        ethnicity = int(raw_data['ethnicity'].item())
        language = int(raw_data['language'].item())
        return dict(jpg=target, txt=prompt, hint=source,name=name,glaucoma_label=glaucoma_label,race=race,gender=gender,ethnicity=ethnicity,language=language)


import json
import cv2
import numpy as np
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, RandomHorizontalFlip, RandomVerticalFlip, Compose,ToPILImage
import torchvision.transforms as F
import os
import pandas as pd
from glob import glob
def random_crop(image,start_h,start_w, crop_height, crop_width):
    if image.shape[0] < crop_height or image.shape[1] < crop_width:
        raise ValueError("Image dimensions should be larger than crop dimensions")
    cropped_image = image[start_h:start_h+crop_height, start_w:start_w+crop_width]

    return cropped_image

class MyDataset(Dataset):
    def __init__(self,args,gen_data = False,file_list = None):
        self.data = []
        
        summary = os.path.join(args.dataset_dir, 'data_summary.csv')
        self.summary_data = pd.read_csv(summary).set_index('filename')
        self.gen_data = gen_data
        self.mask_file = os.path.join(args.dataset_dir,'mask')

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.palette = {0: (0, 0, 0),
           -1: (255, 0, 0),  # 红色
           -2: (0, 0, 255)}  # 蓝色

        if file_list != None:
            self.data = file_list
        else:
            self.data = glob(os.path.join(args.dataset_dir, 'Training', "*.npz"))
            with open(os.path.join(args.dataset_dir,'filter_file.txt'),'r') as f:
                file_list = f.read()
                file_list = file_list.split('\n')
                file_list.remove("")
                file_list = [i.replace('.png','.npz') for i in file_list]
                file_list = set(file_list)
            self.data = [i for i in self.data if os.path.basename(i) in file_list]








    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_file = self.data[idx]
        raw_data = np.load(data_file, allow_pickle=True)
        modified_image = raw_data['slo_fundus']
        modified_image = modified_image.astype(np.uint8)
        modified_image = np.array([modified_image, modified_image, modified_image]).transpose(1,2,0)
        clahe_img = np.zeros_like(modified_image)
        for i in range(3):
            clahe_img[:, :, i] = self.clahe.apply(modified_image[:, :, i])
        # mask
        mask_file = os.path.basename(data_file).replace(".npz",".png")
        mask_file = os.path.join(self.mask_file,mask_file)

        mask_image = Image.open(mask_file)
        mask_image = mask_image.convert('RGB')
        mask_image = np.array(mask_image)


        source = mask_image #664*789
        target = clahe_img #664*789 mask_image

        index = os.path.basename(data_file)
        info = self.summary_data.loc[index]
        prompt = 'Age ' + str(info['age']) + ', ' + ', '.join(info.iloc[2:7].values.tolist())
        prompt += ', ' + info['gpt4_summary']








        crop_height = crop_width = 512
        start_h = np.random.randint(0, source.shape[0] - crop_height+1)
        start_w = np.random.randint(0, source.shape[1] - crop_width+1)
        source = random_crop(source,start_h,start_w, crop_height, crop_width)
        target = random_crop(target,start_h,start_w, crop_height, crop_width)
        if not self.gen_data:
            r = random.randint(1,100)
            if r >= 50:
                source = cv2.flip(source, 1)
                target = cv2.flip(target, 1)
            r = random.randint(1, 100)
            if r >= 50:
                source = cv2.flip(source, 0)
                target = cv2.flip(target, 0)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        # target = resize(target, (256, 256))
        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0
        name =  os.path.basename(data_file).replace(".npz","")
        glaucoma_label = int(raw_data['glaucoma'].item())
        race = int(raw_data['race'].item())
        gender = int(raw_data['gender'].item())
        ethnicity = int(raw_data['ethnicity'].item())
        language = int(raw_data['language'].item())
        return dict(jpg=target, txt=prompt, hint=source,name=name,glaucoma_label=glaucoma_label,race=race,gender=gender,ethnicity=ethnicity,language=language)

