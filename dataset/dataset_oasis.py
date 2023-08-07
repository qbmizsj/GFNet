import os
import torch
import numpy as np 
from glob import glob

import SimpleITK as sitk
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torch.nn as nn


class OASISdataset(Dataset):
    def __init__(self, label_info, img_info, num_ad, type, transform=None):
        self.transform = transform
        self.label_info = label_info
        self.img_path = img_info
        img_list, label_list = self.assign_label(self.img_path, self.label_info, num=num_ad)
        if type == 'train':
            self.patients_list, self.label_list, _, _ = self.split_dataset(img_list, label_list)
        else:
            _, _, self.patients_list, self.label_list = self.split_dataset(img_list, label_list)

    def __getitem__(self, index):
        # OAS30001_ClinicalData_d1106
        # AD_class/oasis/preprocessed/T1/sub-OAS30001_ses-d0129_run-01_T1w.nii.gz
        # AD_class/oasis/preprocessed/T1/sub-OAS30001_ses-d2430_T1w.nii.gz
        patient_dir = self.patients_list[index]
        label = self.label_list[index]
        data_path = os.path.join(self.img_path, patient_dir)

        # print('data_path:', data_path)
        nifti_image = sitk.ReadImage(data_path)
        np_img = sitk.GetArrayFromImage(nifti_image)
        # (182, 218, 182)
        img = torch.Tensor(np_img).permute(2,1,0).unsqueeze(dim=0)
        img = img[:,::2,::2,::2]
        aug_img = img.clone()
        if self.transform is not None:
            aug_img = self.transform(aug_img)
        # 32*3, 32*4, 32*3 --> (2,3,9,10,2,3)
        # 16*6, 16*7, 16*6
        pad_img = nn.ConstantPad3d((2,3,1,2,2,3),0)
        aug_img = pad_img(aug_img)
        img = pad_img(img)
        img = img.type(torch.float32)  

        aug_img = aug_img.type(torch.float32)
        return aug_img, img, label

    def __len__(self):
        return len(self.label_info)


    def assign_label(self, dataset_path, label_path, num):
        dataset = os.listdir(dataset_path)
        df = pd.read_csv(label_path, usecols = ["ADRC_ADRCCLINICALDATA ID", "cdr"], header=0)
        imgs, labels = df["ADRC_ADRCCLINICALDATA ID"], df["cdr"]
        img_info = imgs.tolist()
        np_imgs = np.array(img_info)
        label_info = labels.tolist()
        label_list = []
        img_list = []
        dict = {'0.0':0, '0.5':0, '1.0':0, '2.0':0, '3.0':0}
        # print("dataset:", len(dataset))
        for files in dataset:
            file, _, _ = files.split(".")
            # ['sub-OAS30748_ses-d0219_run-01_T1w', 'nii', 'gz']
            # 'sub-OAS30970_ses-d0238_T1w.nii.gz'
            subject = file[4:4+8]
            id = file[17:17+5]
            patient = subject + '_ClinicalData_' + id
            bound = int(np.argwhere(np_imgs > patient)[0])
            # print(":", bound, img_info[bound], np_imgs[bound], patient, img_info[bound][7]==patient[7], img_info[bound-1][7]==patient[7])
            if (img_info[bound][7]==patient[7] and img_info[bound-1][7]==patient[7]):
                if (label_info[bound]==0 and label_info[bound-1]==0):
                    if dict['0.0']>=num:
                        continue
                    else:
                        label_list.append(0)
                        img_list.append(files)
                        dict['0.0'] += 1
                else:
                    label_list.append(1)
                    img_list.append(files)
                    dict[str(max(label_info[bound], label_info[bound-1]))] += 1
            else:
                for idx in range(2):
                    if np_imgs[bound-idx][7]==patient[7]:
                        if label_info[bound-idx]>0:
                            label_list.append(1)
                            img_list.append(files)
                        else:
                            if dict['0.0']>=num:
                                continue
                            else:
                                label_list.append(0)
                                img_list.append(files)
                        dict[str(label_info[bound-idx])] += 1

        assert len(img_list) == len(label_list)
        # print("label_list.count(1):", label_list.count(1), label_list.count(0))
        return img_list, label_list


    def split_dataset(self, img, label, nfold=10):
        n_patients = len(label)
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
            if i == 0:
                for idx in fold:
                    val_patients_list.append(img[idx])
                    val_label_list.append(label[idx])
            else:
                for idx in fold:
                    train_patients_list.append(img[idx])
                    train_label_list.append(label[idx])
        # print(f"train patients: {len(train_patients_list)}, test patients: {len(val_patients_list)}")

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
