from share import *
import config

# import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
from PIL import Image

from pytorch_lightning import seed_everything
# from annotator.util import resize_image, HWC3
# from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from torchvision import transforms
from torch.utils.data import DataLoader
from eye_dataset import eyeDataset
from tutorial_dataset import MyDataset
import os
from tqdm import tqdm
import uuid
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
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


        print(prompt, a_prompt)
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


model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict('lightning_logs/version_22/checkpoints/epoch=49-step=142895.ckpt', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


# transform = transforms.Compose([
#     transforms.Resize((512, 512)),
#     # transforms.ToTensor(),
# ])

dataset = MyDataset(gen_data=True)
dataloader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=False)


save_dir = './results/new_summary_img'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


count = 0
for i,item in enumerate(tqdm(dataloader)):

    target = item['jpg'].to('cuda')
    prompt = item['txt']
    source = item['hint'].to('cuda')
    output_name = item['name'][0]
    # rs_target = item['rs_target']
    output = process(source, prompt, '', '', 1, 512, 512, ddim_steps=30, guess_mode=False, strength=1.0, scale=9.0, seed=np.random.randint(0, 65536, 1), eta=0)
    for j in range(0, len(output)):
        generated_uuid = str(uuid.uuid4())[:4]
        img = output[j]
        img = transforms.ToPILImage()(img)
        if j == 0:
            img.save(os.path.join(save_dir, f'{output_name}_{generated_uuid}_mask.png'))
            with open(os.path.join(save_dir, f'{output_name}_{generated_uuid}_prompt.txt'), 'w') as f:
                f.write(prompt[0])
        else:
            img.save(os.path.join(save_dir, f'{output_name}_{generated_uuid}_generate.png'))
    




    count += 1
    # if count > 2:
    #     break