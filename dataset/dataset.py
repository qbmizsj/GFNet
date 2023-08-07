import os
import torch
import numpy as np 
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torch.nn as nn
from utils import class_to_num
import matplotlib.pyplot as plt


class ADNIdataset_sp(Dataset):
    def __init__(self, label_path, img_path, type):
        self.label_path = label_path
        self.img_path = img_path
        self.patients_list, self.label_list = self.file_process(label_path)

        if type =='train':
            self.patients_list, self.label_list, _, _  = self.split_dataset(self.patients_list, self.label_list)
        else:
            _, _, self.patients_list, self.label_list = self.split_dataset(self.patients_list, self.label_list)
        # 分类任务不需要sort list，
        

    def __getitem__(self, index):
        
        patient_dir = self.patients_list[index]
        str_label = self.label_list[index]
        # print("index:", index, patient_dir, str_label)
        label = class_to_num(str_label)
        data_path = os.path.join(self.img_path, patient_dir + '.npy')
        np_img = np.load(data_path)
        img = torch.Tensor(np_img).permute(2,1,0).unsqueeze(dim=0)
        img = img[::2,::2,::2]
        mask_img = img.clone()
        # 32*3, 32*3+16, 32*3
        # pad_img = nn.ConstantPad3d((2,3,1,2,2,3),0)
        # pad_img = nn.ConstantPad3d((5,6,3,4,5,6),0)
        pad_img = nn.ConstantPad3d((2,3,9,10,2,3),0)
        mask_img = pad_img(mask_img)
        # when using multi-processing, do not use torch.cuda.FloatTensor
        img = img.type(torch.float32)
        #img = img.repeat(3,1,1,1)        
        mask_img = mask_img.type(torch.float32)
        #mask_img = mask_img.repeat(3,1,1,1)
        return mask_img, img, label


    def __len__(self):
        return len(self.patients_list)


    def file_process(self, path):
        f = open(path, encoding="utf-8")
        df = pd.read_csv(f, usecols = ["filename", "status"], header=0)
        imgs, labels = df["filename"], df["status"]
        img_list = imgs.tolist()
        label_list = labels.tolist()
        return img_list, label_list
    

    def split_dataset(self, patients_dir, patients_label, nfold=10, select=0):
        # patients_dir = os.listdir(self.file_list)
        # patients_label = os.listdir(self.label_list)
        n_patients = len(patients_dir)
        # 给病人排序号
        pid_idx = np.arange(n_patients)
        np.random.seed(123)
        np.random.shuffle(pid_idx)
        n_fold_list = np.array_split(pid_idx, nfold)
        #print(f"split {len(n_fold_list)} folds and every fold have {len(n_fold_list[0])} patients")
        val_patients_list = []
        train_patients_list = []
        val_label_list = []
        train_label_list = []
        for i, fold in enumerate(n_fold_list):
            if i == select:
                for idx in fold:
                    val_patients_list.append(patients_dir[idx])
                    val_label_list.append(patients_label[idx])
            else:
                for idx in fold:
                    train_patients_list.append(patients_dir[idx])
                    train_label_list.append(patients_label[idx])
        #print(f"train patients: {len(train_patients_list)}, test patients: {len(val_patients_list)}")

        return train_patients_list, train_label_list, val_patients_list, val_label_list




if __name__ == '__main__':
    pass
    '''
    label_path = '/data/users/zhangsj/AD_class/lookupcsv/ADNI.csv'
    img_path = '/data/users/zhangsj/AD_class/ADNI_1'
    train_dataset = ADNIdataset(label_path, img_path, type = 'train')
    val_dataset = ADNIdataset(label_path, img_path, type = 'val')
    train_loader = DataLoader(train_dataset, batch_size = 16, drop_last = True, shuffle = 'True')
    val_loader = DataLoader(val_dataset, batch_size = 16, drop_last = True, shuffle = 'False')
    for img, label in train_loader:
        print("label:", label)
    '''
