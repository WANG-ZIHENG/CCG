# import sys
# sys.path.append("../")
from annotator.util import resize_image, HWC3
from glob import glob
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import json
input_path = 'fairvlmed10k/controlnet/target/*'

lines = []
for filename in tqdm(glob(input_path)):
    basename = os.path.basename(filename)
    prompt_file = f"fairvlmed10k/prompt/{basename.replace('.png', '.txt')}"
    if not os.path.exists(prompt_file):
        continue
    with open(prompt_file , 'r') as f:
        prompt = f.read()
    lines.append({"source": f"source/{basename}", "target": f"target/{basename}", "prompt": prompt})

with open('fairvlmed10k/prompt.json', 'w') as f:
    for i in lines:
        json.dump(i, f)
        f.write("\n")



