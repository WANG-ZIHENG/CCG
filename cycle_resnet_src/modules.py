# Resource:
# https://github.com/openai/CLIP/issues/83
# https://github.com/openai/CLIP
# https://github.com/mlfoundations/wise-ft
# https://github.com/LightDXY/FT-CLIP/tree/main
# https://github.com/damian0815/finetune-clip-huggingface/tree/main
import os
import numpy as np
import random
from PIL import Image
import math
import copy
import pandas as pd
import re
from glob import glob
# import clip
clip = None
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from torchvision.models import *
import torch.nn.functional as F

from sklearn.metrics import *
from fairlearn.metrics import *

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

def find_all_files(folder, suffix='npz'):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and os.path.join(folder, f).endswith(suffix)]
    return files

class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model
        
    def forward(self,text):
        return self.model.encode_text(text)
    
class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model
        
    def forward(self,image):
        return self.model.encode_image(image)

def truncate_note(note, max_length=180):

    # truncate the note if it is longer than 77 words, but maintain the word integrity
    if len(note) > max_length:
        note = note[:max_length]
        note = note[:note.rfind(' ')]
    
    return note

def count_number_of_groups(input_dataset):
    instances_on_race = []
    instances_on_gender = []
    instances_on_ethnicity = []
    for file in input_dataset.files:
            npz_path = file
            data = np.load(npz_path)
            instances_on_race.append(data['race'].item())
            instances_on_gender.append(data['gender'].item())
            instances_on_ethnicity.append(data['ethnicity'].item())
    # count the unique number in instances_on_race
    _, numbers_of_race = np.unique(instances_on_race, return_counts=True)
    _, numbers_of_gender = np.unique(instances_on_gender, return_counts=True)
    _, numbers_of_ethnicity = np.unique(instances_on_ethnicity, return_counts=True)
    return numbers_of_race, numbers_of_gender, numbers_of_ethnicity


class fair_vl_med_dataset(torch.utils.data.Dataset):
    def __init__(self, args,dataset_dir='', preprocess=None,use_gen_data=False, files=None, subset='training', text_source='note', summarized_note_file=None, ruleout_unknown=False, exclude_original_label=None, exclude_generated_label=None, extra_train_data_dir=None, balance_attribute=None):
        self.use_gen_data = use_gen_data
        self.preprocess = preprocess
        self.dataset_dir = os.path.join(dataset_dir, "All")
        self.subset = subset
        self.text_source = text_source
        self.ruleout_unknown = ruleout_unknown
        self.exclude_original_label = exclude_original_label
        self.exclude_generated_label = exclude_generated_label
        self.extra_train_data_dir = getattr(args, 'extra_train_data_dir', '') if extra_train_data_dir is None else extra_train_data_dir
        self.balance_attribute = getattr(args, 'balance_attribute', None) if balance_attribute is None else balance_attribute
        # 记录哪些文件来自额外数据目录
        self.extra_data_files = set()
        
        # age 转换函数：<=65 为年轻(0)，>65 为老年(1)
        def convert_age_to_group(age_value):
            """将年龄转换为组别：<=65 为年轻(0)，>65 为老年(1)"""

            age_value = float(age_value)
            if isinstance(age_value, (int, float, np.integer, np.floating)):
                age = float(age_value)
                return 0 if age <= 65 else 1
            else:
                # 如果已经是组别值，直接返回
                return int(age_value)
        
        self.convert_age_to_group = convert_age_to_group


        self.summarized_notes = {}
        self.glucoma_labels = {}  # 存储从CSV读取的glaucoma标签
        # 记录属性标签对应的文本：{filename: {age: xxx, gender: xxx, ...}}
        self.attribute_texts = {}
        # 记录属性文本到数字id的映射：{'gender': {'female': 1, 'male': 0}, ...}
        self.attribute_text2id = {}
        # summarized_note_file is a csv file that contains the summarized notes associated with npz files
        # read the summarized notes from the csv file and construct a dictionary

        # 读取summarized_note_file时，记录每个文件属于train/val/test
        file_to_use = {}
        if self.text_source == 'note' and summarized_note_file != '':
            df = pd.read_csv(os.path.join(dataset_dir, summarized_note_file))
            for index, row in df.iterrows():
                # filename 优先使用显式列名，否则使用第一列
                if 'filename' in df.columns:
                    filename = str(row['filename']).strip()
                else:
                    filename = str(row.iloc[0]).strip()

                # 记录 note 文本
                # 注意：这里假设第 9 列是 note，如果你的 csv 格式有变化，需要相应调整
                if len(row) > 8:
                    self.summarized_notes[filename] = str(row.iloc[8]).strip()

                # 记录 use 列（文件属于 train/val/test）
                file_to_use[filename] = row['use'] if 'use' in df.columns else None

                # 读取 glaucoma 标签（如果 CSV 中有 label 列）
                if 'label' in df.columns:
                    glaucoma_val = str(row['label'])
                    self.glucoma_labels[filename] = 1 if glaucoma_val.lower() == 'glaucoma' else 0

                # 如果是训练集，收集属性标签对应的文本
                if self.subset == 'training':
                    attr_dict = {}
                    for col in ['age', 'gender', 'race', 'ethnicity', 'language', 'maritalstatus']:
                        if col in df.columns:
                            attr_dict[col] = str(row[col])
                    if attr_dict:
                        self.attribute_texts[filename] = attr_dict


        self.files = []
        if files is not None :
            print(f"use {files}")
            files = os.path.join(dataset_dir, files)
                    # check if the split file exists
            with open(files,'r',encoding='utf-8') as f:
                filter = f.read()
                filter = filter.split('\n')
                filter = [i.replace(".png",'.npz') for i in filter]
            files = find_all_files(self.dataset_dir, suffix='npz')
            basename_files = [os.path.basename(i) for i in files]
            files = {os.path.basename(i): i for i in files}
            filter_files = set(filter) & set(basename_files)
            for k,v in list(files.items()):
                if k  in filter_files and file_to_use.get(k) == subset:
                    self.files.append(v)




            # 额外添加sd_gen数据
            li = glob(os.path.join(args.result_dir, "sd_gen*", "*generate*"))
            # li = [os.path.basename(i) for i in li]
            if self.use_gen_data:
                print("使用生成数据")
            for img_path in tqdm(li):
                source_file = re.findall('data_\d+',img_path)[0]
                # num = int(re.findall('data_(\d+)',img_path)[0])

                source_file += ".npz"
                file = os.path.basename(img_path)
                file = file.replace(".png", ".npz")
                self.summarized_notes[file] = self.summarized_notes[source_file]

                out_path = os.path.join(os.path.dirname(img_path),file)
                if self.use_gen_data:

                    self.files.append(out_path)
                else:
                    pass
                npz_path = os.path.join(self.dataset_dir,source_file)
                if  os.path.exists(out_path):
                    continue
                else:
                    data = np.load(npz_path)
                    data = dict(data)
                    img = Image.open(img_path)
                    img = img.convert('L')
                    img_array = np.array(img)
                    data['slo_fundus'] = img_array
                    np.savez(out_path, **data)

        else:
            # df = pd.read_csv(os.path.join(dataset_dir, 'split_files.csv'))
            # self.files = df[df['file_type'] == subset]['filename'].tolist()
            
            self.files = find_all_files(self.dataset_dir, suffix='npz')
            if file_to_use:
                self.files = [i for i in self.files if file_to_use.get(os.path.basename(i)) == subset]

        # 根据glaucoma_label排除原始数据集中的样本（仅对训练集）
        if self.subset == 'training' and self.exclude_original_label is not None:
            filtered_files = []
            removed = 0
            for file in self.files:
                with np.load(file) as data:
                    # 尝试从data中读取glaucoma标签
                    if 'glaucoma' in data:
                        glaucoma_val = int(data['glaucoma'].item())
                    elif 'glaucoma_label' in data:
                        glaucoma_val = int(data['glaucoma_label'].item())
                    else:
                        # 如果data中没有，尝试从self.glucoma_labels中获取
                        file_basename = os.path.basename(file)
                        if file_basename in self.glucoma_labels:
                            glaucoma_val = self.glucoma_labels[file_basename]
                        else:
                            glaucoma_val = None
                if glaucoma_val == self.exclude_original_label:
                    removed += 1
                else:
                    filtered_files.append(file)
            if removed > 0:
                print(f"[fair_vl_med_dataset] Excluded {removed} original training samples with glaucoma_label "
                      f"== {self.exclude_original_label}")
            self.files = filtered_files

        # iterate through the files and remove the ones that has unknown attributes (-1)
        if subset != 'training' or self.ruleout_unknown:
            tmp_files = []
            for file in self.files:
                npz_path = file if os.path.isabs(file) else os.path.join(self.dataset_dir, file)
                data = np.load(npz_path)
                if data['race'].item() != -1 and data['gender'].item() != -1 and data['ethnicity'].item() != -1 and data['language'].item() != -1:
                    tmp_files.append(file)
            self.files = tmp_files

        # Append extra training data if provided
        if self.subset == 'training' and self.extra_train_data_dir:
            # cup-disc threshold 从 args 获取，默认为 0.6
            self.cup_disc_threshold = getattr(args, 'cup_disc_threshold')
            # 每组随机选择数量（用于属性平衡），默认为 2
            self.samples_per_group = int(getattr(args, 'samples_per_group'))
            extra_dir = self.extra_train_data_dir
            if not os.path.isabs(extra_dir):
                extra_dir = os.path.join(dataset_dir, extra_dir)
            if os.path.isdir(extra_dir):
                extra_files = find_all_files(extra_dir, suffix='npz')
                existing_names = set(os.path.basename(path) for path in self.files)
                new_files = [f for f in extra_files if os.path.basename(f) not in existing_names]
                if not new_files:
                    print(f"[fair_vl_med_dataset] No new files found in extra dir: {extra_dir}")
                else:
                    # 根据glaucoma_label排除生成数据中的样本
                    if self.exclude_generated_label is not None:
                        filtered_new_files = []
                        removed = 0
                        for file in new_files:
                            with np.load(file) as data:
                                if 'glaucoma' in data:
                                    glaucoma_val = int(data['glaucoma'].item())
                                elif 'glaucoma_label' in data:
                                    glaucoma_val = int(data['glaucoma_label'].item())
                                else:
                                    # 如果data中没有，尝试从self.glucoma_labels中获取
                                    file_basename = os.path.basename(file)
                                    if file_basename in self.glucoma_labels:
                                        glaucoma_val = self.glucoma_labels[file_basename]
                                    else:
                                        glaucoma_val = None
                            if glaucoma_val == self.exclude_generated_label:
                                removed += 1
                            else:
                                filtered_new_files.append(file)
                        if removed > 0:
                            print(f"[fair_vl_med_dataset] Excluded {removed} generated training samples with glaucoma_label "
                                  f"== {self.exclude_generated_label}")
                        new_files = filtered_new_files
                    
                    if new_files:
                        print(f"[fair_vl_med_dataset] Adding {len(new_files)} extra training samples from {extra_dir}")
                        #


                        # new_files 是你刚获取到的 npz 文件列表
                        # 例如 ["data_05646_gen_4090_scale_0_ratio_0.37.npz", ...]

                        grouped = defaultdict(list)

                        # 按 data_id 分组
                        data_id_pattern = re.compile(r"data_(\d+)_")

                        for f in new_files:
                            m = data_id_pattern.search(f)
                            if m:
                                data_id = m.group(1)
                                grouped[data_id].append(f)
                            else:
                                raise

                        # 每组随机取 n 张（由 args.samples_per_group 控制）
                        n =  int(getattr(self, 'samples_per_group'))
                        selected_files = []
                        for data_id, files in grouped.items():
                            if len(files) <= n:
                                selected_files.extend(files)
                            else:
                                selected_files.extend(random.sample(files, n))

                        # 记录
                        self.extra_data_files.update(selected_files)
                        self.files.extend(selected_files)


                        # # 记录额外数据文件
                        # self.extra_data_files.update(new_files)
                        # self.files.extend(new_files)
                    else:
                        print(f"[fair_vl_med_dataset] No extra training samples remaining after filtering")
            else:
                raise
        
        # 统计原数据和生成数据中的 glaucoma 标签数量（0/1）
        original_label_0 = 0
        original_label_1 = 0
        generated_label_0 = 0
        generated_label_1 = 0

        for file in self.files:
            try:
                data = np.load(file)
                is_extra_data = file in self.extra_data_files

                # 优先从 npz 中读取 glaucoma / glaucoma_label，其次从 CSV 读取
                if 'glaucoma_label' in data:
                    glaucoma_label = int(data['glaucoma_label'].item())
                elif 'glaucoma' in data:
                    glaucoma_label = int(data['glaucoma'].item())
                else:
                    file_basename = os.path.basename(file)
                    if file_basename in self.glucoma_labels:
                        glaucoma_label = self.glucoma_labels[file_basename]
                    else:
                        continue

                if is_extra_data:
                    # 生成数据
                    if glaucoma_label == 0:
                        generated_label_0 += 1
                    elif glaucoma_label == 1:
                        generated_label_1 += 1
                else:
                    # 原始数据
                    if glaucoma_label == 0:
                        original_label_0 += 1
                    elif glaucoma_label == 1:
                        original_label_1 += 1
            except Exception:
                continue

        print(f"[fair_vl_med_dataset] Glaucoma label statistics for {self.subset} set:")
        print("  Original data:")
        print(f"    - glaucoma_label 0: {original_label_0}")
        print(f"    - glaucoma_label 1: {original_label_1}")
        print(f"    - Total: {original_label_0 + original_label_1}")
        print("  Generated data (extra_train_data_dir):")
        print(f"    - glaucoma_label 0: {generated_label_0}")
        print(f"    - glaucoma_label 1: {generated_label_1}")
        print(f"    - Total: {generated_label_0 + generated_label_1}")
        print("  Overall:")
        print(f"    - glaucoma_label 0: {original_label_0 + generated_label_0}")
        print(f"    - glaucoma_label 1: {original_label_1 + generated_label_1}")
        print(f"    - Total samples: {len(self.files)}")

        # ======================
        # 属性平衡逻辑：使用生成数据平衡指定属性的各个组别
        # ======================
        if self.subset == 'training' and self.balance_attribute and self.extra_train_data_dir:
            valid_attributes = ['gender', 'race', 'age']
            if self.balance_attribute not in valid_attributes:
                print(f"[fair_vl_med_dataset] Warning: balance_attribute '{self.balance_attribute}' is not valid. Valid options: {valid_attributes}")
            else:
                print(f"[fair_vl_med_dataset] Balancing data by attribute: {self.balance_attribute}")
                
                # 统计原始数据中各个组别的数量
                original_group_counts = {}
                original_files_by_group = {}
                
                for file in self.files:
                    if file in self.extra_data_files:
                        continue  # 跳过已添加的生成数据
                    try:
                        data = np.load(file)
                        if self.balance_attribute == 'age':
                            # age 需要转换：<=65 为年轻(0)，>65 为老年(1)
                            age_value = data['age'].item()
                            attr_value = self.convert_age_to_group(age_value)
                        else:
                            attr_value = int(data[self.balance_attribute].item())
                        if attr_value not in original_group_counts:
                            original_group_counts[attr_value] = 0
                            original_files_by_group[attr_value] = []
                        original_group_counts[attr_value] += 1
                        original_files_by_group[attr_value].append(file)
                    except Exception as e:
                        continue
                
                print(f"[fair_vl_med_dataset] Original data group counts for {self.balance_attribute}:")
                for group_id, count in sorted(original_group_counts.items()):
                    print(f"  - Group {group_id}: {count}")
                
                # 找到数量最多的组别作为目标数量
                if original_group_counts:
                    max_count = max(original_group_counts.values())
                    target_count = max_count
                    print(f"[fair_vl_med_dataset] Target count per group: {target_count}")
                    
                    # 从生成数据中选择数据来平衡各个组别
                    # 首先收集所有可用的生成数据，按属性分组
                    available_gen_files_by_group = {}
                    
                    if os.path.isdir(self.extra_train_data_dir):
                        extra_dir = self.extra_train_data_dir
                        if not os.path.isabs(extra_dir):
                            extra_dir = os.path.join(dataset_dir, extra_dir)
                        
                        if os.path.isdir(extra_dir):
                            all_extra_files = find_all_files(extra_dir, suffix='npz')
                            
                            for file in all_extra_files:
                                # 检查是否已被排除
                                if self.exclude_generated_label is not None:
                                    try:
                                        data = np.load(file)
                                        if 'glaucoma' in data:
                                            glaucoma_val = int(data['glaucoma'].item())
                                        elif 'glaucoma_label' in data:
                                            glaucoma_val = int(data['glaucoma_label'].item())
                                        else:
                                            continue
                                        if glaucoma_val == self.exclude_generated_label:
                                            continue
                                    except Exception:
                                        continue
                                
                                # 读取属性值
                                try:
                                    data = np.load(file)
                                    # 生成数据的属性值应该从原始数据中继承，但这里我们假设生成数据也有该属性
                                    # 如果生成数据没有该属性，我们需要从源文件中获取
                                    if self.balance_attribute in data:
                                        if self.balance_attribute == 'age':
                                            # age 需要转换：<=65 为年轻(0)，>65 为老年(1)
                                            age_value = data['age'].item()
                                            attr_value = self.convert_age_to_group(age_value)
                                        else:
                                            attr_value = int(data[self.balance_attribute].item())
                                    else:
                                        # 如果生成数据没有该属性，尝试从文件名或其他方式推断
                                        # 这里我们跳过没有该属性的生成数据
                                        continue
                                    
                                    if attr_value not in available_gen_files_by_group:
                                        available_gen_files_by_group[attr_value] = []
                                    available_gen_files_by_group[attr_value].append(file)
                                except Exception:
                                    continue
                            
                            # 为每个组别选择需要的生成数据
                            selected_gen_files = []
                            for group_id in original_group_counts.keys():
                                current_count = original_group_counts[group_id]
                                needed_count = max(0, target_count - current_count)
                                
                                if needed_count > 0 and group_id in available_gen_files_by_group:
                                    available_files = available_gen_files_by_group[group_id]
                                    # 随机选择需要的数量
                                    random.shuffle(available_files)
                                    selected = available_files[:needed_count]
                                    selected_gen_files.extend(selected)
                                    print(f"[fair_vl_med_dataset] Selected {len(selected)} generated samples for group {group_id} (needed {needed_count})")
                            
                            # 更新文件列表
                            if selected_gen_files:
                                # 移除之前添加的所有生成数据
                                self.files = [f for f in self.files if f not in self.extra_data_files]
                                self.extra_data_files.clear()
                                
                                # 添加平衡后的生成数据
                                self.extra_data_files.update(selected_gen_files)
                                self.files.extend(selected_gen_files)
                                
                                print(f"[fair_vl_med_dataset] Added {len(selected_gen_files)} generated samples for balancing")
                                
                                # 重新统计平衡后的组别数量
                                balanced_group_counts = {}
                                for file in self.files:
                                    try:
                                        data = np.load(file)
                                        if self.balance_attribute == 'age':
                                            # age 需要转换：<=65 为年轻(0)，>65 为老年(1)
                                            age_value = data['age'].item()
                                            attr_value = self.convert_age_to_group(age_value)
                                        else:
                                            attr_value = int(data[self.balance_attribute].item())
                                        balanced_group_counts[attr_value] = balanced_group_counts.get(attr_value, 0) + 1
                                    except Exception:
                                        continue
                                
                                print(f"[fair_vl_med_dataset] Balanced group counts for {self.balance_attribute}:")
                                for group_id, count in sorted(balanced_group_counts.items()):
                                    print(f"  - Group {group_id}: {count}")
                            else:
                                print(f"[fair_vl_med_dataset] No suitable generated data found for balancing")

        # ======================
        # 训练集：构建并打印「属性文本 -> 数字标签」映射
        # 例如：gender: {'female': 1, 'male': 0}
        # ======================
        if self.subset == 'training' and self.attribute_texts:
            mapping = {}
            for file in self.files:
                # 只对原始数据构建映射（生成数据属性全设为0，没有文本）
                if file in self.extra_data_files:
                    continue
                fname = os.path.basename(file)
                if fname not in self.attribute_texts:
                    continue
                try:
                    data = np.load(file)
                except Exception:
                    continue

                attr_text = self.attribute_texts[fname]

                # gender
                if 'gender' in attr_text and 'gender' in data:
                    num_val = int(data['gender'].item())
                    text_val = str(attr_text['gender'])
                    mapping.setdefault('gender', {})[text_val] = num_val

                # race
                if 'race' in attr_text and 'race' in data:
                    num_val = int(data['race'].item())
                    text_val = str(attr_text['race'])
                    mapping.setdefault('race', {})[text_val] = num_val

                # ethnicity (hispanic)
                if 'ethnicity' in attr_text and 'ethnicity' in data:
                    num_val = int(data['ethnicity'].item())
                    text_val = str(attr_text['ethnicity'])
                    mapping.setdefault('ethnicity', {})[text_val] = num_val

                # language
                if 'language' in attr_text and 'language' in data:
                    num_val = int(data['language'].item())
                    text_val = str(attr_text['language'])
                    mapping.setdefault('language', {})[text_val] = num_val




                # maritalstatus（如果 npz 中也有）
                if 'maritalstatus' in attr_text and 'maritalstatus' in data:
                    num_val = int(data['maritalstatus'].item())
                    text_val = str(attr_text['maritalstatus'])
                    mapping.setdefault('maritalstatus', {})[text_val] = num_val

            self.attribute_text2id = mapping
            print("[fair_vl_med_dataset] Attribute text -> id mapping for training set:")
            print(self.attribute_text2id)
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npz_path =  self.files[idx]
        data = np.load(npz_path)
        is_extra_data = npz_path in self.extra_data_files
        
        try:
            slo_fundus = data['slo_fundus'].astype(np.float32) # original size: (664, 512)
        except:
            print(npz_path)
        slo_fundus = self.preprocess(Image.fromarray(slo_fundus).convert('RGB'))

        if self.subset == 'training':
            if self.text_source == 'note':
                note = self.summarized_notes.get(os.path.basename(self.files[idx]), '').strip()

                note = truncate_note(note)
                # token = clip.tokenize(note)
                # token = token.squeeze()
                token = 0
            elif self.text_source == 'label':
                # 对于额外数据，优先依据 cup_disc_ratio（若存在）来判断 glaucoma_label，
                # 否则尝试读取 glaucoma_label 或 glaucoma 字段，最后回退到 CSV 中的记录
                if is_extra_data:
                    ratio_value = None
                    if 'cup_disc_ratio' in data:
                        ratio_value = float(data['cup_disc_ratio'].item())
                    if ratio_value is not None:
                        # 使用阈值判断（大于阈值为 1，否则为 0）
                        glaucoma_label = 1 if ratio_value > self.cup_disc_threshold else 0
                    else:
                        glaucoma_label = int(data['glaucoma'].item())
                else:
                    # 如果'glaucoma'不在data里面，就从self.glucoma_labels里面获取
                    file_basename = os.path.basename(self.files[idx])
                    if 'glaucoma' in data:
                        glaucoma_label = int(data['glaucoma'].item())
                    elif file_basename in self.glucoma_labels:
                        glaucoma_label = self.glucoma_labels[file_basename]
                    else:
                        # 如果都找不到，默认设为0
                        glaucoma_label = 0
                if glaucoma_label == 1:
                    note = 'A photo of glaucoma'
                else:
                    note = 'A photo of non-glaucoma'
                # token = clip.tokenize(note)
                # token = token.squeeze()
                token = 0
        else:
            note = 'A photo of non-glaucoma'
            # neg_token = clip.tokenize(note)

            note = 'A photo of glaucoma'
            # pos_token = clip.tokenize(note)

            # concatenate two tensors together, the final tensor will be at size of 2, 77
            # token = torch.cat((neg_token, pos_token), dim=0)
            token = 0

        # extract glaucoma label from npz file or summarized_notes
        # 对于额外数据，只读取 slo_fundus 和 glaucoma_label，其他属性设为0
        if is_extra_data:
            ratio_value = None
            if 'cup_disc_ratio' in data:
                ratio_value = float(data['cup_disc_ratio'].item())

            if ratio_value is not None:
                # 使用阈值判断（大于阈值为 1，否则为 0）
                glaucoma_label = 1 if ratio_value > self.cup_disc_threshold else 0
            else:
                glaucoma_label = int(data['glaucoma'].item())
            race = int(data['race'].item())
            gender = int(data['gender'].item())
            hispanic = int(data['ethnicity'].item())
            language = int(data['language'].item())
            # 读取 age 并转换为组别：<=65 为年轻(0)，>65 为老年(1)
            if 'age' in data:
                age_value = data['age'].item()
                age_group = self.convert_age_to_group(age_value)
            else:
                age_group = 0  # 如果没有 age 信息，默认设为0
        else:
            # 正常数据，读取所有属性
            # 如果'glaucoma'不在data里面，就从self.glucoma_labels里面获取
            file_basename = os.path.basename(self.files[idx])
            if 'glaucoma' in data:
                glaucoma_label = int(data['glaucoma'].item())
            elif file_basename in self.glucoma_labels:
                glaucoma_label = self.glucoma_labels[file_basename]
            else:
                # 如果都找不到，默认设为0
                glaucoma_label = 0
            race = int(data['race'].item())
            gender = int(data['gender'].item())
            hispanic = int(data['ethnicity'].item())
            language = int(data['language'].item())
            # 读取 age 并转换为组别：<=65 为年轻(0)，>65 为老年(1)
            if 'age' in data:
                age_value = data['age'].item()
                age_group = self.convert_age_to_group(age_value)
            else:
                age_group = 0  # 如果没有 age 信息，默认设为0
        # merge all labels together into a single tensor at size of 6 (添加了 age)
        label_and_attributes = torch.tensor([glaucoma_label, race, gender, hispanic, language, age_group])
        return slo_fundus, token, label_and_attributes,glaucoma_label,npz_path

class fair_vl_group_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir='', preprocess=None, files=None, subset='Training', text_source='note', summarized_note_file=None, attribute='race', thegroup=0):
        self.preprocess = preprocess
        self.dataset_dir = os.path.join(dataset_dir, subset)
        self.subset = subset
        self.text_source = text_source

        self.summarized_notes = {}
        # summarized_note_file is a csv file that contains the summarized notes associated with npz files
        # read the summarized notes from the csv file and construct a dictionary
        if self.subset == 'Training' and self.text_source == 'note' and summarized_note_file != '':
            df = pd.read_csv(os.path.join(dataset_dir, summarized_note_file))
            
            for index, row in df.iterrows():
                self.summarized_notes[row.iloc[0].strip()] = row.iloc[2].strip()
        
        # check if the split file exists
        if files is not None and subset=='Training':
            print(f"use {files}")
            files = os.path.join(dataset_dir, files)
            with open(files,'r',encoding='utf-8') as f:
                filter = f.read()
                filter = filter.split('\n')
                filter = [i.replace(".png",'.npz') for i in filter]
            files = find_all_files(self.dataset_dir, suffix='npz')
            basename_files = [os.path.basename(i) for i in files]
            files = {os.path.basename(i): i for i in files}
            filter_files = set(filter) & set(basename_files)
            for k,v in list(files.items()):
                if k not in filter_files:
                    del files[k]
            self.files = list(files.values())

            # 额外添加sd_gen数据
            li = glob(os.path.join(dataset_dir, "sd_gen", "*generate.png"))
            li = [os.path.basename(i) for i in li]
            li = [[int(i.split("_")[0]),i] for i in li]
            li = sorted(li, key=lambda x: x[0])
            li = [i[1] for i in li]

            test_li = glob(os.path.join(dataset_dir, "Test", "*.npz"))
            test_li = [os.path.basename(i) for i in test_li]
            test_li = list(set(filter) & set(test_li))
            test_li = [[int(re.findall("\d+",i)[0]),i] for i in test_li ]
            test_li = sorted(test_li, key=lambda x: x[0])
            test_li = [i[1] for i in test_li]
            li_map = {k:v for k,v in zip(li,test_li)}
            for k,v in li_map.items():
                npz_path = os.path.join(dataset_dir, "Test", v)
                img_path = os.path.join(dataset_dir, "sd_gen", k)
                out_path = img_path.replace(".png", '.npz')
                basename = os.path.basename(out_path)
                # self.summarized_notes[basename] = self.summarized_notes[v]
                # self.files.append(out_path)
                if  os.path.exists(out_path):
                    continue
                else:
                    data = np.load(npz_path)
                    data = dict(data)
                    img = Image.open(img_path)
                    img = img.convert('L')
                    img_array = np.array(img)
                    data['slo_fundus'] = img_array
                    np.savez(out_path, **data)
        else:
            # df = pd.read_csv(os.path.join(dataset_dir, 'split_files.csv'))
            # self.files = df[df['file_type'] == subset]['filename'].tolist()
            self.files = find_all_files(self.dataset_dir, suffix='npz')

        # df = pd.read_csv(os.path.join(dataset_dir, 'split_files.csv'))
        # self.files = df[df['file_type'] == subset]['filename'].tolist()
        # self.files = files

        # iterate through the files and remove the ones that has unknown attributes (-1)
        if subset != 'Training':
            tmp_files = []
            for file in self.files:
                npz_path = os.path.join(self.dataset_dir, file)
                data = np.load(npz_path)
                if data['race'].item() != -1 and data['gender'].item() != -1 and data['ethnicity'].item() != -1 and data['language'].item() != -1:
                    tmp_files.append(file)
            self.files = tmp_files

        tmp_files = []
        for file in self.files:
            npz_path = os.path.join(self.dataset_dir, file)
            data = np.load(npz_path)

            group = int(data[attribute].item())
            if group == thegroup:
                tmp_files.append(file)
        self.files = tmp_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npz_path = os.path.join(self.dataset_dir, self.files[idx])
        data = np.load(npz_path)
        slo_fundus = data['slo_fundus'].astype(np.float32) # original size: (664, 512)
        slo_fundus = self.preprocess(Image.fromarray(slo_fundus))

        if self.subset == 'Training':
            if self.text_source == 'note':

                note = self.summarized_notes[os.path.basename(self.files[idx])].strip()

                note = truncate_note(note)
                token = clip.tokenize(note)
                token = token.squeeze()
            elif self.text_source == 'label':
                glaucoma_label = int(data['glaucoma'].item())
                if glaucoma_label == 1:
                    note = 'A photo of glaucoma'
                else:
                    note = 'A photo of non-glaucoma'
                token = clip.tokenize(note)
                token = token.squeeze()
        else:
            note = 'A photo of non-glaucoma'
            neg_token = clip.tokenize(note)

            note = 'A photo of glaucoma'
            pos_token = clip.tokenize(note)

            # concatenate two tensors together, the final tensor will be at size of 2, 77
            token = torch.cat((neg_token, pos_token), dim=0)

        # extract glaucoma label from npz file
        glaucoma_label = int(data['glaucoma'].item())
        race = int(data['race'].item())
        gender = int(data['gender'].item())
        hispanic = int(data['ethnicity'].item())
        language = int(data['language'].item())
        # merge all labels together into a single tensor at size of 4
        label_and_attributes = torch.tensor([glaucoma_label, race, gender, hispanic, language])

        return slo_fundus, token, label_and_attributes

def endless_loader(dataloader):
    while True:
        for data in dataloader:
            yield data

class image_title_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir='', preprocess=None, files=None, subset='train'):
        self.preprocess = preprocess
        self.files = files
        self.dataset_dir = dataset_dir
        self.subset = subset

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npz_path = os.path.join(self.dataset_dir, self.files[idx])
        data = np.load(npz_path)
        slo_fundus = data['slo_fundus'].astype(np.float32)
        slo_fundus = self.preprocess(Image.fromarray(slo_fundus))

        if self.subset == 'train':
            note = truncate_note(data['note'].item().strip())
            token = clip.tokenize(note)
            token = token.squeeze()
        else:
            note = 'A photo of non-glaucoma'
            neg_token = clip.tokenize(note)

            note = 'A photo of glaucoma'
            pos_token = clip.tokenize(note)

            # concatenate two tensors together, the final tensor will be at size of 2, 77
            token = torch.cat((neg_token, pos_token), dim=0)

        # extract glaucoma label from npz file
        glaucoma_label = int(data['glaucoma'].item())
        race = int(data['race'].item())
        gender = int(data['gender'].item())
        hispanic = int(data['hispanic'].item())
        # merge all labels together into a single tensor at size of 4
        label_and_attributes = torch.tensor([glaucoma_label, race, gender, hispanic])


        return slo_fundus, token, label_and_attributes

class Adversary_Net(nn.Module):

    def __init__(self, n_sensitive, n_hidden=32):
        super(Adversary_Net, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_hidden, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, n_sensitive),
        )

    def forward(self, x):
        return self.network(x)

def compute_vl_prob(img_feats, class_txt_feats):
    # img_feats: [batch_size, 512]
    # class_txt_feats: [batch_size, num_class, 512]

    all_logits = []
    for i in range(class_txt_feats.shape[1]):
        similarity = (img_feats @ class_txt_feats[:, i, :].T)
        # extract the diagonal of the matrix
        logits = similarity.diag()
        all_logits.append(logits)

    all_logits = torch.stack(all_logits, dim=1)

    # compute the probability by applying softmax along the second dimension
    vl_prob = torch.softmax(all_logits, dim=1)

    return vl_prob, all_logits

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if len(output.shape) == 1:
        acc = np.sum((output >= 0.5).astype(float) == target)/target.shape[0]
        return acc.item()
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, dim=1)
        target = target.view(batch_size, 1).repeat(1, maxk)
        
        correct = (pred == target)
  
        topk_accuracy = []
        for k in topk:
            accuracy = correct[:, :k].float().sum().item()
            accuracy /= batch_size # [0, 1.]
            topk_accuracy.append(accuracy)
        
        return topk_accuracy[0]

def compute_auc(pred_prob, y, num_classes=2):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    if num_classes == 2:
        fpr, tpr, thresholds = roc_curve(y, pred_prob)
        auc_val = auc(fpr, tpr)
    elif num_classes > 2:
        y_onehot = num_to_onehot(y, num_classes)
        auc_val = roc_auc_score(y_onehot, pred_prob, average='macro', multi_class='ovr')

    return auc_val

def auc_score(pred_prob, y):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    if np.unique(y).shape[0]>2:
        AUC = roc_auc_score(y, pred_prob, multi_class='ovr')
    else:
        fpr, tpr, thresholds = roc_curve(y, pred_prob)
        AUC = auc(fpr, tpr)
    
    return AUC

def num_to_onehot(nums, num_to_class):
    nums = nums.astype(int)
    n_values = num_to_class
    onehot_vec = np.eye(n_values)[nums]
    return onehot_vec

def prob_to_label(pred_prob):
    # Find the indices of the highest probabilities for each sample
    max_prob_indices = np.argmax(pred_prob, axis=1)

    # Create one-hot vectors for each sample
    one_hot_vectors = np.zeros_like(pred_prob)
    one_hot_vectors[np.arange(len(max_prob_indices)), max_prob_indices] = 1

    return one_hot_vectors

def numeric_to_one_hot(y, num_classes=None):
    y = np.asarray(y, dtype=np.int32)

    if num_classes is None:
        num_classes = np.max(y) + 1
    
    one_hot_array = np.zeros((len(y), num_classes))
    one_hot_array[np.arange(len(y)), y] = 1
    
    return one_hot_array

def multiclass_demographic_parity(pred_prob, y, attrs):

    pred_one_hot = prob_to_label(pred_prob)

    gt_one_hot = numeric_to_one_hot(y)

    scores = []
    for i in range(pred_one_hot.shape[1]):
        tmp_score = demographic_parity_difference(pred_one_hot[:,i],
                                gt_one_hot[:,i],
                                sensitive_features=attrs)

        scores.append(tmp_score)

    avg_score = np.mean(scores)
        
    return avg_score

def multiclass_equalized_odds(pred_prob, y, attrs):

    pred_one_hot = prob_to_label(pred_prob)

    gt_one_hot = numeric_to_one_hot(y)

    scores = []
    for i in range(pred_one_hot.shape[1]):
        tmp_score = equalized_odds_difference(pred_one_hot[:,i],
                            gt_one_hot[:,i],
                            sensitive_features=attrs)

        scores.append(tmp_score)

    avg_score = np.mean(scores)
        
    return avg_score

def multiclass_demographic_parity_(pred_prob, y, attrs):

    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    attrs_set = np.unique(attrs)
    y_pred = np.argmax(pred_prob, axis=1)

    mc_dpd = 0
    for i in range(pred_prob.shape[1]):
        tmp_preds = (y_pred==i).astype(int)
        tmp_not_preds = 1 - tmp_preds

        dp_by_attrs = []
        for j in attrs_set:
            idx = attrs==j
            tmp = np.abs(tmp_preds.mean().item() - tmp_preds[idx].mean().item()) + np.abs(tmp_not_preds.mean().item() - tmp_not_preds[idx].mean().item())
            dp_by_attrs.append(tmp)
            print(tmp)
        mc_dpd += np.mean(dp_by_attrs).item()

    mc_dpd = mc_dpd / pred_prob.shape[1]
        
    return mc_dpd

def auc_score_multiclass(pred_prob, y, num_of_class=3, eps=0.01):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    sensitivity_at_diff_specificity = [-1]*4
    y_onehot = num_to_onehot(y, num_of_class)
    fpr, tpr, thresholds = roc_curve(y_onehot.ravel(), pred_prob.ravel())
    for i in range(len(fpr)):
        cur_fpr = fpr[i]
        cur_tpr = tpr[i]
        if np.abs(cur_fpr-0.2) <= eps:
            sensitivity_at_diff_specificity[0] = cur_tpr
        if np.abs(cur_fpr-0.15) <= eps:
            sensitivity_at_diff_specificity[1] = cur_tpr
        if np.abs(cur_fpr-0.1) <= eps:
            sensitivity_at_diff_specificity[2] = cur_tpr
        if np.abs(cur_fpr-0.05) <= eps:
            sensitivity_at_diff_specificity[3] = cur_tpr
    AUC = auc(fpr, tpr)
    
    return AUC, sensitivity_at_diff_specificity

def equity_scaled_accuracy(output, target, attrs, alpha=1.):
    es_acc = 0
    if len(output.shape) >= 2:
        overall_acc = np.sum(np.argmax(output, axis=1) == target)/target.shape[0]
    else:
        overall_acc = np.sum((output >= 0.5).astype(float) == target)/target.shape[0]
    tmp = 0
    identity_wise_perf = []
    identity_wise_num = []
    for one_attr in np.unique(attrs).astype(int):
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]

        if len(pred_group.shape) >= 2:
            acc = np.sum(np.argmax(pred_group, axis=1) == gt_group)/gt_group.shape[0]
        else:
            acc = np.sum((pred_group >= 0.5).astype(float) == gt_group)/gt_group.shape[0]

        identity_wise_perf.append(acc)
        identity_wise_num.append(gt_group.shape[0])

    for i in range(len(identity_wise_perf)):
        tmp += np.abs(identity_wise_perf[i]-overall_acc)
    es_acc = (overall_acc / (alpha*tmp + 1))
    
    return es_acc

def equity_scaled_AUC(output, target, attrs, alpha=1., num_classes=2):
    es_auc = 0
    tmp = 0
    identity_wise_perf = []
    identity_wise_num = []
    
    if num_classes == 2:
        fpr, tpr, thresholds = roc_curve(target, output)
        overall_auc = auc(fpr, tpr)
    elif num_classes > 2:
        y_onehot = num_to_onehot(target, num_classes)
        overall_auc = roc_auc_score(y_onehot, output, average='macro', multi_class='ovr')

    for one_attr in np.unique(attrs).astype(int):
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]

        if num_classes == 2:
            fpr, tpr, thresholds = roc_curve(gt_group, pred_group)
            group_auc = auc(fpr, tpr)
        elif num_classes > 2:
            y_onehot = num_to_onehot(gt_group, num_classes)
            group_auc = roc_auc_score(y_onehot, pred_group, average='macro', multi_class='ovr')
        
        identity_wise_perf.append(group_auc)
        identity_wise_num.append(gt_group.shape[0])

    for i in range(len(identity_wise_perf)):
        tmp += np.abs(identity_wise_perf[i]-overall_auc)
    es_auc = (overall_auc / (alpha*tmp + 1))

    return es_auc

def evalute_perf_by_attr(preds, gts, attrs=None, num_classes=2):

    esaccs_by_attrs = []
    esaucs_by_attrs = []
    aucs_by_attrs = []
    dpds = []
    dprs = []
    eods = []
    eors = []
    for i in range(attrs.shape[0]):
        attr = attrs[i,:]

        es_acc = equity_scaled_accuracy(preds, gts, attr)
        esaccs_by_attrs.append(es_acc)
        es_auc = equity_scaled_AUC(preds, gts, attr, num_classes=num_classes)
        esaucs_by_attrs.append(es_auc)

        aucs_by_group = []
        elements = np.unique(attr).astype(int)
        for e in elements:
            aucs_by_group.append( compute_auc(preds[attr == e], gts[attr == e], num_classes=num_classes) )
        aucs_by_attrs.append(aucs_by_group)
        pred_labels = (preds >= 0.5).astype(float)
        if num_classes == 2:
            dpd = demographic_parity_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            dpr = demographic_parity_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eod = equalized_odds_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eor = equalized_odds_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
        elif num_classes > 2:
            dpd = multiclass_demographic_parity(preds, gts, attr)
            dpr = 0
            eod = multiclass_equalized_odds(preds, gts, attr)
            eor = 0

        dpds.append(dpd)
        eods.append(eod)

    return esaccs_by_attrs, esaucs_by_attrs, aucs_by_attrs, dpds, eods


def evalute_comprehensive_perf(preds, gts, attrs=None, num_classes=2):

    esaccs_by_attrs = []
    esaucs_by_attrs = []
    aucs_by_attrs = []
    dpds = []
    dprs = []
    eods = []
    eors = []
    between_group_disparity = []

    overall_acc = accuracy(preds, gts, topk=(1,))
    overall_auc = compute_auc(preds, gts, num_classes=num_classes)
    # 计算混淆矩阵
    pred_labels = (preds >= 0.5).astype(float)
    tn, fp, fn, tp = confusion_matrix(gts, pred_labels).ravel()
    # 计算specificity
    specificity = tn / (tn + fp)
    # 计算sensitivity
    sensitivity = tp / (tp + fn)
    # 计算precision
    precision = tp / (tp + fp)
    # 计算F1 score
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    # 计算 MCC (Matthews Correlation Coefficient)
    mcc = matthews_corrcoef(gts, pred_labels)
    # 计算 QWK (Quadratic Weighted Kappa)
    qwk = cohen_kappa_score(gts, pred_labels, weights='quadratic')

    for i in range(attrs.shape[0]):
        attr = attrs[i,:]

        es_acc = equity_scaled_accuracy(preds, gts, attr)
        esaccs_by_attrs.append(es_acc)
        es_auc = equity_scaled_AUC(preds, gts, attr, num_classes=num_classes)
        esaucs_by_attrs.append(es_auc)

        aucs_by_group = []
        elements = np.unique(attr).astype(int)
        for e in elements:
            aucs_by_group.append( compute_auc(preds[attr == e], gts[attr == e], num_classes=num_classes) )
        aucs_by_attrs.append(aucs_by_group)
        std_disparity, max_disparity = compute_between_group_disparity(aucs_by_group, overall_auc)
        between_group_disparity.append([std_disparity, max_disparity])

        pred_labels = (preds >= 0.5).astype(float)
        if num_classes == 2:
            dpd = demographic_parity_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            dpr = demographic_parity_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eod = equalized_odds_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eor = equalized_odds_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
        elif num_classes > 2:
            dpd = multiclass_demographic_parity(preds, gts, attr)
            dpr = 0
            eod = multiclass_equalized_odds(preds, gts, attr)
            eor = 0

        dpds.append(dpd)
        eods.append(eod)

    return overall_acc, esaccs_by_attrs, overall_auc, esaucs_by_attrs, aucs_by_attrs, dpds, eods, between_group_disparity, specificity, sensitivity, f1, precision, mcc, qwk

def evalute_comprehensive_perf_(preds, gts, attrs=None, num_classes=2):

    esaccs_by_attrs = []
    esaucs_by_attrs = []
    aucs_by_attrs = []
    dpds = []
    dprs = []
    eods = []
    eors = []
    between_group_disparity = []

    overall_auc = compute_auc(preds, gts, num_classes=num_classes)

    for i in range(attrs.shape[0]):
        attr = attrs[i,:]

        es_acc = equity_scaled_accuracy(preds, gts, attr)
        esaccs_by_attrs.append(es_acc)
        es_auc = equity_scaled_AUC(preds, gts, attr, num_classes=num_classes)
        esaucs_by_attrs.append(es_auc)

        aucs_by_group = []
        elements = np.unique(attr).astype(int)
        for e in elements:
            aucs_by_group.append( compute_auc(preds[attr == e], gts[attr == e], num_classes=num_classes) )
        aucs_by_attrs.append(aucs_by_group)
        std_disparity, max_disparity = compute_between_group_disparity_half(aucs_by_group, overall_auc)
        between_group_disparity.append([std_disparity, max_disparity])

        pred_labels = (preds >= 0.5).astype(float)
        if num_classes == 2:
            dpd = demographic_parity_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            dpr = demographic_parity_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eod = equalized_odds_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eor = equalized_odds_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
        elif num_classes > 2:
            dpd = multiclass_demographic_parity(preds, gts, attr)
            dpr = 0
            eod = multiclass_equalized_odds(preds, gts, attr)
            eor = 0

        dpds.append(dpd)
        eods.append(eod)

    return esaccs_by_attrs, esaucs_by_attrs, aucs_by_attrs, dpds, eods, between_group_disparity

def compute_between_group_disparity(auc_list, overall_auc):
    return np.std(auc_list) / overall_auc, (np.max(auc_list)-np.min(auc_list)) / overall_auc

def compute_between_group_disparity_half(auc_list, overall_auc):
    return np.std(auc_list) / np.abs(overall_auc-0.5), (np.max(auc_list)-np.min(auc_list)) / np.abs(overall_auc-0.5)


class EMA_confidence():
    def __init__(self, confidence, decay):
        self.confidence = confidence
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self,name,value):
        if name not in self.shadow:
            self.shadow[name] = value


    def update(self):
        for key, (value,label) in self.confidence.items():
            new_average = (1.0 - self.decay) * value + self.decay * self.shadow[key][0]
            self.shadow[key] = [new_average,label]

    def apply_shadow(self):
        for  key, (value,label) in self.confidence.items():
            self.backup[key] = [value,label]
            self.confidence[key] = self.shadow[key]

    def restore(self):
        for key, (value,label) in self.confidence.items():

            self.confidence[key] = self.backup[key]
        self.backup = {}


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}




