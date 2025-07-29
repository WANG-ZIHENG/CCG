from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from eye_dataset import eyeDataset
from torchvision import transforms

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# # Transform
# transform = transforms.Compose([
#     transforms.Resize((512, 512)),
#     transforms.ToTensor(),
# ])


# Misc
# dataset = MyDataset()
dataset = eyeDataset(data_path='./data', split='train', prompt_type='summary', source_transform=None, target_transform=None)
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(default_root_dir='./train_logs/summary', gpus=1, precision=32, callbacks=[logger], max_epochs=15)


# Train!
trainer.fit(model, dataloader)
