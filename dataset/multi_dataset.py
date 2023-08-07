import os
import torch
import numpy as np 
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torch.nn as nn
from utils import class_to_num


class MIL_ADNIdataset(Dataset):
    def __init__(self, data):
        self.info_list = data

    def __getitem__(self, index):
        mask_img, img, label = self.info_list[index]
        img = img.unsqueeze(dim=0)
        mask_img = mask_img.unsqueeze(dim=0)

        return mask_img, img, label

    def __len__(self):
        return len(self.info_list)


if __name__ == '__main__':
    label_path = '/home/zhang_istbi/zhangsj/ACGF/ADNI.csv'
    img_path = '/home/zhang_istbi/zhangsj/ACGF/ADNI_1'
    transform = transforms.Compose([
                transforms.ToTensor()
                ])
    dataset = MIL_ADNIdataset(label_path, img_path, 16, transform)
    # 切patch在main函数操作
    print('len(dataset):', len(dataset))
