import torch
from torch.utils.data import Dataset
import numpy as np


class bagdata(Dataset):
    def __init__(self, imgs, labels):
        self.img_list = imgs
        self.label_list = labels

        print("label_list[:10]2:", labels[:10])
        '''
        if type == 'train':
            train_patients_list, train_label_list, _, _ = self.split_dataset(imgs, labels)
        else:
            _, _, val_patients_list, val_label_list = self.split_dataset(imgs, labels)
        '''

    def __getitem__(self, index):
        bag = self.img_list[index]
        label = self.label_list[index]
        return bag, label


    def __len__(self):
        return len(self.label_list)


    def split_dataset(self, img, label, nfold=10):
        n_patients = label.shape
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
        #print(f"train patients: {len(train_patients_list)}, test patients: {len(val_patients_list)}")

        return train_patients_list, train_label_list, val_patients_list, val_label_list


if __name__ == '__main__':
    pass