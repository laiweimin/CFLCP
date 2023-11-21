import copy

import numpy as np
import os
import random

from collections import defaultdict

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from utils.dataset_utils import split_data, save_file
from os import path
from scipy.io import loadmat
from PIL import Image
from torch.utils.data import DataLoader,random_split
from torchvision.datasets import ImageFolder



class OfficeCaltech10Dataset(data.Dataset):
    def __init__(self, data_folder, transform=None):
        # 初始化数据集
        self.data = []  # 包含图像路径和标签的列表
        self.transform = transform

        # 从data_folder中加载数据，将图像路径和标签添加到self.data中

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        # 加载图像
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return img, label


data_path = "data/office_caltech_10/"

# Allocate data to usersz``
def generate_Office10(dir_path):

    # 创建数据集
    # dataset = OfficeCaltech10Dataset(data_path, transform=transform)

    # 创建数据加载器
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # train_loader, test_loader = digit5_dataset_read(dir_path, '1')
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    domains = ['amazon', 'caltech', 'dslr', 'webcam']
    train_dataloader=[]
    test_dataloader=[]
    posi=0
    for d in domains:
        # train_loader, test_loader = digit5_dataset_read(root, d)
        # train_loader = torch.utils.data.DataLoader(train_loader,
        #                                    batch_size=24,
        #                                    sampler=torch.utils.data.sampler.SubsetRandomSampler(list(random.sample(range(25000), 500))))
        img_dataset=ImageFolder(root=data_path + d, transform=transform)
        img_dataset_len=len(img_dataset)
        client_len=int(img_dataset_len*0.8/5)
        img_dataset_split=random_split(img_dataset,[client_len,client_len,client_len,client_len,client_len,img_dataset_len-5*client_len])
        train_dataloader.extend([(posi+pos, DataLoader(img_dataset_split[pos],batch_size=16)) for pos in range(5)])
        test_dataloader.extend([(posi+pos, DataLoader(img_dataset_split[5],batch_size=16)) for pos in range(5)])
        posi=posi+5
    # print(1)
    return train_dataloader,test_dataloader

def generate_shift_Office10(dir_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    domains = ['amazon', 'caltech', 'dslr', 'webcam']
    train_dataloader = []
    test_dataloader = []
    posi = 0
    for d in domains:
        img_dataset = ImageFolder(root=data_path + d, transform=transform)
        img_dataset_len = len(img_dataset)
        client_len = int(img_dataset_len * 0.8 / 5)

        all_numbers = list(range(10))

        for pos in range(5):
            class_indices = random.sample(all_numbers, 2)
            filtered_indices = [i for i in range(len(img_dataset)) if img_dataset.targets[i] == class_indices[0]]
            filtered_indices1 = [i for i in range(len(img_dataset)) if img_dataset.targets[i] == class_indices[1]]
            class_train_len = int(len(filtered_indices)*0.8)
            class_train_len1 = int(len(filtered_indices1)*0.8)
            if class_train_len>40:
                num_samples=30
            else:
                num_samples=int(class_train_len/2)+1
            if class_train_len1 > 40:
                num_samples1 = 30
            else:
                num_samples1 = int(class_train_len1 / 2) + 1
            subset_indices = list(random.sample(filtered_indices[:class_train_len], num_samples))
            subset_indices1 = list(random.sample(filtered_indices1[:class_train_len1], num_samples1))
            subset_indices.extend(subset_indices1)
            test_ind=copy.deepcopy(filtered_indices[class_train_len:])
            test_ind.extend(copy.deepcopy(filtered_indices1[class_train_len1:]))
            train_dataloader.extend([(posi + pos, DataLoader(img_dataset,
                                                                              batch_size=16,
                                                                              sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                                                  subset_indices)))])


            test_dataloader.extend([(posi + pos, torch.utils.data.DataLoader(img_dataset,
                                                                             batch_size=16,
                                                                             sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                                                 test_ind)))])
        posi = posi + 5



        # img_dataset_split = random_split(img_dataset, [client_len, client_len, client_len, client_len, client_len,
        #                                                img_dataset_len - 5 * client_len])
        # train_dataloader.extend([(posi + pos, DataLoader(img_dataset_split[pos], batch_size=16)) for pos in range(5)])
        # posi = posi + 5
        # test_dataloader.append(torch.utils.data.DataLoader(img_dataset_split[5],
        #                                                    batch_size=16))
    # print(1)
    return train_dataloader, test_dataloader




if __name__ == "__main__":
    generate_Office10(data_path)