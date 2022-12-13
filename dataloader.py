import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from   torchvision import transforms
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import wandb
import csv

import cv2 as cv, numpy as np
import os as os

dir_img = './images'
dir_label = './label'

class ImageFile(object):
    def __init__(self, phase="train"):
        self.phase = phase
    
    def _get_filenames_(self, file_dir):
        list =os.listdir(file_dir)
        list.sort(key=lambda x: int(x[0:-4]))
        return list
    
    def _list_abspath(self, file_dir, file_list):
        return [os.path.join(file_dir, file_name) for file_name in file_list]

    def _get_labels_(self, dir_label):
        with open(dir_label + '/label_.csv', 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
            eyes = [row[1] for row in data]
            hair = [row[2] for row in data]
            mouth_open = [row[3] for row in data]
            gender = [row[4] for row in data]
            glasses = [row[5] for row in data]
        return [eyes, hair, mouth_open, gender, glasses]
    
    
class TrainDataset(ImageFile):
    def __init__(self):
        super(TrainDataset, self).__init__(phase="train")
        
        self.img_dir = dir_img
        self.label_dir = dir_label
        
        self.img_list = self._get_filenames_(self.img_dir)

        self.labels = self._get_labels_(self.label_dir)
        self.img = self._list_abspath(self.img_dir, self.img_list)
        
    
    def __len__(self):
        return len(self.img)
    

class ToTensor(object):
    """
    Convert ndarrays in dataset to Tensors with normalization.
    """
    def __init__(self, phase="train"):
        self.mean = torch.tensor([0.5, 0.5, 0.5]).view(3,1,1) # 
        self.std = torch.tensor([0.5,  0.5, 0.5]).view(3,1,1) #
        self.phase = phase

    def __call__(self, sample):
        
        if self.phase == "train":
            # convert BGR images to RGB
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            img = np.resize(sample['img'], (64,64,3))
            img = img[:,:,::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['img'] = torch.from_numpy(img).sub_(self.mean).div_(self.std)
            # del sample['image_name']
        sample['mouth_open'] = torch.tensor(sample['mouth_open']).to(torch.long)
        sample['gender'] = torch.tensor(sample['gender']).to(torch.long)
        sample['glasses'] = torch.tensor(sample['glasses']).to(torch.long)
        sample['label'] = torch.from_numpy(sample['label']).to(torch.long)             

        return sample

def color2code(color):
    code = np.zeros(6)
    code[color - 1] = 1
    return code

def bool2code(bool_):
    code = np.zeros(2)
    code[bool_] = 1
    return code

def feature2code(sample):
    eyes = color2code(sample['eyes'])
    hair = color2code(sample['hair'])
    mouth_open = bool2code(sample['mouth_open'])
    gender = bool2code(sample['gender'])
    glasses = bool2code(sample['glasses'])
    code = np.concatenate((eyes, hair, mouth_open, gender, glasses))
    return code

class dataLoader(Dataset):
    def __init__(self, data, phase = "train"):
        self.phase = phase
        self.img = data.img
        self.eyes = data.labels[0]
        self.hair = data.labels[1]
        self.mouth_open = data.labels[2]
        self.gender = data.labels[3]
        self.glasses = data.labels[4]
        self.img_list = data.img_list
        
        self.transform = {
            'train':
                transforms.Compose([ToTensor()]),
            'val':
                transforms.Compose(ToTensor()),
            'test':
                transforms.Compose(ToTensor())}[phase]

        
    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, idx):
        img = cv.imread(self.img[idx])
        name = self.img_list[idx]
        eyes = int(self.eyes[idx])
        hair = int(self.hair[idx])
        mouth_open = int(self.mouth_open[idx])
        gender = int(self.gender[idx])
        glasses = int(self.glasses[idx])
        sample = {'img': img, 'eyes': eyes, 'hair': hair, 'mouth_open': mouth_open, 'gender': gender, 'glasses': glasses, 'name': name}  
        label = feature2code(sample)
        sample.update({'label': label})
        sample = self.transform(sample)
        return sample

if __name__ == '__main__':
    classifier_train = TrainDataset()
    dataset = dataLoader(classifier_train)

    print(dataset.__getitem__(100)['name'], ': ', dataset.__getitem__(100)['label'])