from share import *
import sys
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import argparse

parser = argparse.ArgumentParser(description='FairCLIP Training/Fine-Tuning')

parser.add_argument('--dataset_dir', default='/root/autodl-tmp/fairvlmed10k', type=str)
# Configs
resume_path = './models/control_sd21_ini.ckpt'
batch_size = 3
logger_freq = 30000
learning_rate = 1e-5
sd_locked = False
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


args = parser.parse_args()
# Misc
dataset = MyDataset(args)
dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq,log_images_kwargs={"ddim_steps":50})
save_checkpoint = pl.callbacks.ModelCheckpoint(every_n_epochs=3, save_top_k=1)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger,save_checkpoint])


# Train!
trainer.fit(model, dataloader,ckpt_path='/root/autodl-tmp/ControlNet_new/lightning_logs/version_11/checkpoints/epoch=1-step=7021.ckpt')
# trainer.fit(model, dataloader)
