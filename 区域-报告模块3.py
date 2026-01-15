import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import monai.transforms as transforms
from monai.data import PersistentDataset
import nibabel as nib
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from functools import partial
import torch.nn.functional as F
import tqdm
import random
import pickle


class RadGenomeDataset_Train(PersistentDataset):
    def __init__(self, text_tokenizer, data_folder, mask_folder, csv_file, cache_dir, max_region_size=10,
                 max_img_size=1, image_num=32, region_num=33, max_seq=2048, resize_dim=500, voc_size=32000,
                 force_num_frames=True):
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            text_tokenizer,
        )

        special_token = {
            "additional_special_tokens": ["<image>", "</image>", "<region>", "</region>"]}

        self.image_padding_tokens = []
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image" + str(i * image_num + j) + ">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append(
                    "<image" + str(i * image_num + j) + ">")
            self.image_padding_tokens.append(image_padding_token)

        self.region_padding_tokens = []
        for i in range(max_region_size):
            region_padding_tokens = ""
            for j in range(region_num):
                region_token = "<region" + str(i * region_num + j) + ">"
                region_padding_tokens = region_padding_tokens + region_token
                special_token["additional_special_tokens"].append(
                    "<region" + str(i * region_num + j) + ">")
            self.region_padding_tokens.append(region_padding_tokens)

        self.text_tokenizer.add_special_tokens(
            special_token
        )
        self.text_tokenizer.pad_token_id = 0
        self.text_tokenizer.bos_token_id = 1
        self.text_tokenizer.eos_token_id = 2

        self.voc_size = voc_size
        self.max_seq = max_seq
        self.data_folder = data_folder
        self.mask_folder = mask_folder

        self.accession_to_sentences = self.load_accession_sentences(csv_file)
        self.paths = []
        self.samples = self.prepare_samples()
        self.target_size = (256, 256, 64)

        def threshold(x):
            return x > -1000

        self.region_transform = transforms.Compose([
            transforms.CropForeground(select_fn=threshold),
            transforms.Resize(spatial_size=self.target_size),
            transforms.ToTensor()
        ])
        self.image_transform = transforms.Compose([
            transforms.CropForegroundd(keys=['img', 'seg'], source_key='img', select_fn=threshold),
            transforms.Resized(keys=['img', 'seg'], spatial_size=self.target_size),
            transforms.ToTensord(keys=['img', 'seg'])
        ])
        self.mask_img_to_tensor = partial(self.mask_nii_img_to_tensor, region_transform=self.region_transform,
                                          image_transform=self.image_transform)
        super().__init__(data=self.samples, transform=None, cache_dir=cache_dir)

    def load_accession_sentences(self, xlsx_file):
        df = pd.read_csv(xlsx_file)
        df_grouped = df.groupby('Volumename')

        accession_to_sentences = {}
        for accession, group in df_grouped:
            sentences = {}
            for i, row in group.iterrows():
                if pd.isna(row['Anatomy']):
                    anatomy_key = 'whole'
                else:
                    anatomy_key = row['Anatomy']
                sentences[anatomy_key] = row['Sentence']
            accession_to_sentences[accession] = sentences
        return accession_to_sentences

    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        cache_file = os.path.join(current_file_dir, 'train_samples.pkl')

        if os.path.exists(cache_file):
            samples = pickle.load(open(cache_file, 'rb'))
        else:
            for patient_folder in tqdm.tqdm(patient_folders):
                accession_folders = glob.glob(os.path.join(patient_folder, '*'))

                for accession_folder in accession_folders:
                    nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))
                    for nii_file in nii_files:
                        accession_number = nii_file.split("/")[-1]

                        if accession_number not in self.accession_to_sentences:
                            continue

                        single_sample = {}
                        volume_name = accession_number.split(".")[0]
                        mask_path = os.path.join(self.mask_folder, 'seg_' + volume_name)
                        single_sample['image'] = nii_file

                        flag = False
                        for region in REGIONS:
                            if region in self.accession_to_sentences[accession_number]:
                                mask_file = os.path.join(mask_path, region + '.nii.gz')
                                region_report = self.accession_to_sentences[accession_number][region]
                                single_sample[region] = [mask_file, region_report]
                                flag = True
                        if not flag:
                            continue

                        samples.append(single_sample)
                        self.paths.append(nii_file)

            with open(cache_file, 'wb') as f:
                pickle.dump(samples, f)

        print('Number of samples: ', len(samples))

        return samples

    def __len__(self):
        return len(self.samples)

    def mask_nii_img_to_tensor(self, img_path, mask_paths, region_transform, image_transform):
        img_data = nib.load(img_path).get_fdata()

        mask_img_tensors = {}
        flag = False
        masks = []
        mask_keys = []
        for key, mask_path in mask_paths.items():

            mask_data = nib.load(mask_path).get_fdata()
            masks.append(mask_data)
            mask_keys.append(key)

            if np.sum(mask_data) == 0:
                continue

            mask_img = img_data * mask_data

            mask_img[mask_data == 0] = -1024
            mask_img = mask_img[np.newaxis, ...]

            tensor = region_transform(mask_img)

            hu_min, hu_max = -1000, 200
            tensor = torch.clamp(tensor, hu_min, hu_max)

            tensor = (((tensor + 400) / 600)).float()

            tensor = tensor.repeat(3, 1, 1, 1)

            tensor = tensor.unsqueeze(0)

            mask_img_tensors[key] = tensor
            flag = True

        if not flag:
            print('No mask: ', img_path)

        img_data = img_data[np.newaxis, ...]
        masks_data = np.stack(masks, axis=0)
        tensors = image_transform({'img': img_data, 'seg': masks_data})

        img_tensor = tensors['img']
        img_tensor = torch.clamp(img_tensor, hu_min, hu_max)
        img_tensor = (((img_tensor + 400) / 600)).float()
        img_tensor = img_tensor.repeat(3, 1, 1, 1)
        img_tensor = img_tensor.unsqueeze(0)
        mask_img_tensors['image'] = img_tensor

        masks_tensor = tensors['seg']
        mask_tensors = {}
        for i, key in enumerate(mask_keys):
            mask_tensors[key] = masks_tensor[i].unsqueeze(0)

        return mask_img_tensors, mask_tensors

