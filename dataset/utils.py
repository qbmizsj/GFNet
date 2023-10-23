from torch.nn import functional as F
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset, TensorDataset
import torch.nn as nn
import pandas as pd
import os
from sklearn.metrics import precision_recall_curve, average_precision_score,roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def class_to_num(type):
    label_sorted = ['NL', 'AD']
    num_classes = len(label_sorted)
    class_num_dict = dict(zip(label_sorted, range(num_classes)))
    return class_num_dict[type]


def assign_label(dataset_path, label_path, num):
    dataset = os.listdir(dataset_path)
    df = pd.read_csv(label_path, usecols = ["ADRC_ADRCCLINICALDATA ID", "cdr"], header=0)
    imgs, labels = df["ADRC_ADRCCLINICALDATA ID"], df["cdr"]
    img_info = imgs.tolist()
    np_imgs = np.array(img_info)
    label_info = labels.tolist()
    label_list = []
    img_list = []
    dict = {'0.0':0, '0.5':0, '1.0':0, '2.0':0, '3.0':0}
    print("dataset:", len(dataset))
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
    print("label_list.count(1):", label_list.count(1), label_list.count(0))
    return img_list, label_list



def split_dataset(args, dataset):
    size = len(dataset)
    indice = list(range(size))
    split = int(np.floor(0.9*size))
    if args.shuffle_dataset:
        np.random.seed(args.seed)
        np.random.shuffle(indice)
    train_indice, test_indice = indice[0:split], indice[split:]
    return train_indice, test_indice


def sensitivity(y_pred, y_true):
	CM = confusion_matrix(y_true, y_pred) 

    # 切片操作，获取每一个类别各自的 tn, fp, tp, fn
	tn_sum = CM[0, 0] # True Negative
	fp_sum = CM[0, 1]
	tp_sum = CM[1, 1] # True Positive
	fn_sum = CM[1, 0] # False Negative
	Condition_negative = tp_sum + fn_sum + 1e-5
	sensitivity = tp_sum / Condition_negative
    
	return sensitivity

def specificity(y_pred, y_true):
	CM = confusion_matrix(y_true, y_pred) 

	tn_sum = CM[0, 0] # True Negative
	fp_sum = CM[0, 1] # False Positive

	tp_sum = CM[1, 1] # True Positive
	fn_sum = CM[1, 0] # False Negative

	Condition_negative = tn_sum + fp_sum + 1e-5
	Specificity = tn_sum / Condition_negative

	return Specificity


def auc(labels, predictions):
    # Sort predictions and labels in descending order based on predictions
    sorted_indices = sorted(range(len(predictions)), key=lambda k: predictions[k], reverse=True)
    sorted_labels = [labels[i] for i in sorted_indices]

    tp = 0  # True positives
    fp = 0  # False positives
    auc = 0  # Area under the curve

    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp

    total_pos = sorted_labels.count(1)
    total_neg = sorted_labels.count(0)
    auc /= (total_pos * total_neg)

    return auc


def acc(true_labels, predicted_labels):
    correct = 0
    total = len(true_labels)
    for true, pred in zip(true_labels, predicted_labels):
        if true == pred:
            correct += 1
    return correct / total


def f1(true_labels, predicted_labels):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for true, pred in zip(true_labels, predicted_labels):
        if true == 1 and pred == 1:
            true_positives += 1
        elif true == 0 and pred == 1:
            false_positives += 1
        elif true == 1 and pred == 0:
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives + 1e-5)
    recall = true_positives / (true_positives + false_negatives + 1e-5)

    return 2 * ((precision * recall) / (precision + recall + 1e-5))

def sen(true_labels, predicted_labels):
    # so called recall
    true_positives = 0
    false_negatives = 0

    for true, pred in zip(true_labels, predicted_labels):
        if true == 1 and pred == 1:
            true_positives += 1
        elif true == 1 and pred == 0:
            false_negatives += 1

    return true_positives / (true_positives + false_negatives + 1e-5)

def spc(true_labels, predicted_labels):
    true_negatives = 0
    false_positives = 0

    for true, pred in zip(true_labels, predicted_labels):
        if true == 0 and pred == 0:
            true_negatives += 1
        elif true == 0 and pred == 1:
            false_positives += 1

    return true_negatives / (true_negatives + false_positives + 1e-5)


def calc_loss_lb(z1, aug1, label, temperature: float = 0.1, pos_only: bool = False):
    ############### redefine pos_mask, add the pair with same label to pos_mask ###############
    '''
    		input: label
    		match instances with same label to construct index set_i={idx_j: j=1,...,k, label(instance_i) = label(instance_j)}
    		original positive pair: (i, i+batch)
    		new one:
    						(i, i+batch)
    						(i, idx_1), ..., (i, idx_j)
    						(i, idx_1+batch), ..., (i, idx_j+batch)
    '''
    # print("label\n:", label)
    device = z1.device
    b = z1.size(0)
    z = torch.cat((z1, aug1), dim=0)
    # 对每个实例的向量做标准化
    z = F.normalize(z, dim=-1)

    # 等价于转置，主对角线表示自己跟自己的相似度
    logits = torch.einsum("if, jf -> ij", z, z) / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    # 对角线置0,其他元素做一个偏置，没有影响
    logits = logits - logits_max.detach()

    # positive mask are matches i, j (i from aug1, j from aug2), where i == j and matches j, i
    pos_mask = torch.zeros((2 * b, 2 * b), dtype=torch.bool, device=device)
    # 标注positive pair 的位置

    pos_mask[:, b:].fill_diagonal_(True)
    pos_mask[b:, :].fill_diagonal_(True)
    '''
    label = torch.cat((label, label), dim=0).bool()
    label = label.to(device)
    label_ad = label
    ad_idx = torch.where(label_ad==1)[0]
    # print('ad_idx:', ad_idx)
    pos_mask[ad_idx] = label_ad
    # use the `~` or `logical_not()` operator instead.
    label_nc = ~label
    nc_idx = torch.where(label_nc==1)[0]
    pos_mask[nc_idx] = label_nc
    print("b pos_mask\n:", pos_mask)
    '''
    label_mask = torch.zeros((b, b), dtype=torch.bool, device=device)
    label = label.bool()
    label_ad = label
    ad_idx = torch.where(label_ad==1)[0]
    label_mask[ad_idx] = label_ad
    # use the `~` or `logical_not()` operator instead.
    label_nc = ~label
    nc_idx = torch.where(label_nc==1)[0]
    label_mask[nc_idx] = label_nc
    # 只用了pos_mask的维数，主对角线置0，其余为1
    pos_mask[:b, :b] = label_mask
    pos_mask[b:, b:] = label_mask
    logit_mask = torch.ones_like(pos_mask, device=device).fill_diagonal_(0)

    exp_logits = torch.exp(logits) * logit_mask

    log_prob = logits if pos_only else logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positives
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
    # print("pos_mask.sum(1):", pos_mask.sum(1))

    loss = -mean_log_prob_pos.mean()
    
    return loss



def calc_loss(z1, aug1, hard=True, temperature: float = 0.1, pos_only: bool = False):
    ############### redefine pos_mask, add the pair with same label to pos_mask ###############
    '''
    		input: label
    		match instances with same label to construct index set_i={idx_j: j=1,...,k, label(instance_i) = label(instance_j)}
    		original positive pair: (i, i+batch)
    		new one:
    						(i, i+batch)
    						(i, idx_1), ..., (i, idx_j)
    						(i, idx_1+batch), ..., (i, idx_j+batch)
    '''

    device = z1.device
    b = z1.size(0)
    z = torch.cat((z1, aug1), dim=0)
    # 对每个实例的向量做标准化
    z = F.normalize(z, dim=-1)

    # 等价于转置，主对角线表示自己跟自己的相似度
    logits = torch.einsum("if, jf -> ij", z, z) / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    # 对角线置0,其他元素做一个偏置，没有影响
    logits = logits - logits_max.detach()

    # positive mask are matches i, j (i from aug1, j from aug2), where i == j and matches j, i
    pos_mask = torch.zeros((2 * b, 2 * b), dtype=torch.bool, device=device)

    # 标注positive pair 的位置

    pos_mask[:, b:].fill_diagonal_(True)
    pos_mask[b:, :].fill_diagonal_(True)
    # 只用了pos_mask的维数，主对角线置0，其余为1
    # logit_mask = torch.ones_like(pos_mask, device=device).fill_diagonal_(0)
    if hard:
        logit_mask = torch.ones_like(pos_mask, device=device).fill_diagonal_(0)
    else:
        cell = torch.zeros((b, b), dtype=torch.bool, device=device)
        cell.fill_diagonal_(True)
        logit_mask = torch.zeros((2 * b, 2 * b), dtype=torch.bool, device=device)
        logit_mask[:b, b:] = ~cell
        logit_mask[b:, :b] = ~cell
        
    exp_logits = torch.exp(logits) * logit_mask

    log_prob = logits if pos_only else logits - torch.log(exp_logits.sum(1, keepdim=True))
    # compute mean of log-likelihood over positives
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)

    loss = -mean_log_prob_pos.mean()
    
    return loss


def max_calc_loss(z1, z2, temperature: float = 0.1, pos_only: bool = False):
    ''' calculate maximize similarity between all sample, including its aug pair '''

    device = z1.device

    b = z1.size(0)
    z = torch.cat((z1, z2), dim=0)
    z = F.normalize(z, dim=-1)

    # 等价于转置，主对角线表示自己跟自己的相似度
    logits = torch.einsum("if, jf -> ij", z, z) / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    # 对角线置0,其他元素做一个偏置，没有影响
    logits = logits - logits_max.detach()

    # all matches excluding the main diagonal
    logit_mask = torch.ones_like(logits, device=device).fill_diagonal_(0)
    
    exp_logits = torch.exp(logits) * logit_mask
    # exp_logits越小越好
    log_prob = torch.log(exp_logits.sum(1, keepdim=True)).mean() 

    return loss


class embedding_evaluation():
    def __init__(self, classifier):
        self.classifier = classifier

    def get_prediction(self, zm, y_true):
        if np.isnan(zm).any():
            print("Has NaNs ... ignoring them")
            zm = np.nan_to_num(zm)
            
        self.classifier.fit(zm, np.squeeze(y_true))
        zm_raw = self.classifier.predict(zm)
        return np.expand_dims(zm_raw, axis=1)

    def calc_eval(zm, y_true):
        img, label = zm.detach().cpu().numpy(), y_true.detach().cpu().numpy()
        y_pred = self.get_prediction(img, label)
        eval_dict = {}
        eval_dict['acc'] = accuracy_score(label, y_pred)
        eval_dict['sen'] = sensitivity(label, y_pred)
        eval_dict['spe'] = specificity(label, y_pred)
        eval_dict['f1'] = f1_score(label, y_pred)
        return eval_dict


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


def split_patch_ds(patient_info, img_path, ps, transform):
    img_list, label_list = file_process(patient_info)
    assert len(img_list)== len(label_list)
    num = len(img_list)
    dataset = []
    for idx in range(num):
        # not in
        if img_list[idx] not in ['ADNI_002_S_0295']: # in ['ADNI_002_S_0295', 'ADNI_002_S_0413', 'ADNI_002_S_0559', 'ADNI_002_S_0619']: # 
            str_label = label_list[idx]
            label = class_to_num(str_label)
            data_path = os.path.join(img_path, img_list[idx] + '.npy')
            np_img = np.load(data_path)
            if transform is not None:
                img = transform(np_img)
                # transform后已经是tensor，不需要再额外送到cuda，数据在loader中提取的时候，会送到cuda
                img = img.permute(1,2,0).type(torch.FloatTensor)
                img = img[::2,::2,::2]
                # img: torch.Size([181, 181, 217])
                # img: torch.Size([91, 109, 91])
            # 生成pador
            pad_img = nn.ConstantPad3d((2,3,10,9,2,3),0)
            # pad_img = nn.ConstantPad3d((3,2,1,2,3,2),0)
            img = pad_img(img)
            img_size = img.shape
            num_d = img_size[2] // ps
            num_w = img_size[1] // ps 
            num_h = img_size[0] // ps
            for d in range(num_d):
                for w in range(num_w):
                    for h in range(num_h):
                        img_patch = img[ps*h: ps*(h+1), ps*w: ps*(w+1), ps*d: ps*(d+1)]
                        dataset.append([img_patch, label])

    return dataset


def split_patch_T(patient_info, img_path, ds_size, ps, transform):
    img_list, label_list = file_process(patient_info)
    assert len(img_list)== len(label_list)
    num = len(img_list)
    print("num:", num)
    dataset = []
    pad_img = nn.ConstantPad3d((2,3,10,9,2,3),0)
    num_d, num_w, num_h = ds_size[0] // ps,  ds_size[1] // ps, ds_size[2] // ps
    for idx in range(num):
        # not in
        if img_list[idx] not in ['ADNI_002_S_0295']: # in ['ADNI_002_S_0295', 'ADNI_002_S_0413', 'ADNI_002_S_0559', 'ADNI_002_S_0619']: # 
            str_label = label_list[idx]
            label = class_to_num(str_label)
            data_path = os.path.join(img_path, img_list[idx] + '.npy')
            np_img = np.load(data_path)
            img = torch.Tensor(np_img).permute(2,1,0)
            aug_img = img.clone()
            
            if transform is not None:  
                aug_img = aug_img.unsqueeze(dim=0)[:,:,:,:]
                aug_img = transform(aug_img)
                aug_img = aug_img.squeeze(dim=0)

            img = img[::2,::2,::2]
            aug_img = aug_img[::2,::2,::2]
            img = pad_img(img)   
            aug_img = pad_img(aug_img)    

            for d in range(num_d):
                for w in range(num_w):
                    for h in range(num_h):
                        aug_img_patch = aug_img[ps*h: ps*(h+1), ps*w: ps*(w+1), ps*d: ps*(d+1)]
                        img_patch = img[ps*h: ps*(h+1), ps*w: ps*(w+1), ps*d: ps*(d+1)]
                        dataset.append([aug_img_patch, img_patch, label])

    return dataset


def file_process(path):
    f = open(path, encoding="utf-8")
    df = pd.read_csv(f, usecols = ["filename", "status"], header=0)
    imgs, labels = df["filename"], df["status"]
    img_list = imgs.tolist()
    label_list = labels.tolist()
    info = [img_list, label_list]
    return info


def patch_to_pred(z_m, label, bag_size):
    pred_list = []
    label_list = []
    b = z_m.size
    print('b:', b)
    num = b // bag_size
    assert b % bag_size == 0
    for idx in range(num):
        bag = z_m[idx*bag_size:(idx+1)*bag_size]
        pred = max(bag)
        pred_list.append(pred)
        label_list.append(label[idx*bag_size])
    
    return pred_list, label_list


def calc_ts_eval(outputs, label, num_for_test):
    output = outputs.clone()
    b, dim = output.shape
    output = output.reshape(int(b/num_for_test), num_for_test, dim)
    if type == 'mean':
        output = output.mean(dim=0)
    else:
        output = output.sum(dim=1)
    output = torch.argmax(output, dim=1)
    output = output.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    eval_dict = {}

    eval_dict['acc'] = acc(label, output)
    #eval_dict['sen'] = sen(label, output)
    #eval_dict['spe'] = spc(label, output)
    eval_dict['f1'] = f1(label, output)
    #eval_dict['auc'] = auc(label, output)
    return eval_dict


def calc_eval(outputs, label):
    # sf = nn.Softmax()
    output = outputs.clone()
    output = torch.argmax(output, dim=1)
    output = output.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    eval_dict = {}
    '''
    eval_dict['acc'] = accuracy_score(label, output)
    eval_dict['sen'] = sensitivity(label, output)
    eval_dict['spe'] = specificity(label, output)
    eval_dict['f1'] = f1_score(label, output)
    '''
    eval_dict['acc'] = acc(label, output)
    #eval_dict['sen'] = sen(label, output)
    #eval_dict['spe'] = spc(label, output)
    eval_dict['f1'] = f1(label, output)
    #eval_dict['auc'] = auc(label, output)
    return eval_dict


def calc_patch_eval(output, label, batch_size, num_for_test, type):
    b = output.shape[0]
    left = b/num_for_test
    if left!=batch_size:
        # 5, 155, 31
        batch_size = int(left)
    split_predictions = torch.split(output, batch_size)
    predictions = torch.stack(split_predictions, dim=0)
    if type == 'mean':
        output = predictions.mean(dim=0)
    else:
        output = predictions.sum(dim=0)
    output = torch.argmax(output, dim=1)
    output = output.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    eval_dict = {}
    '''
    eval_dict['acc'] = accuracy_score(label, output)
    eval_dict['sen'] = sensitivity(label, output)
    eval_dict['spe'] = specificity(label, output)
    eval_dict['f1'] = f1_score(label, output)
    '''
    eval_dict['acc'] = acc(label, output)
    #eval_dict['sen'] = sen(label, output)
    #eval_dict['spe'] = spc(label, output)
    eval_dict['f1'] = f1(label, output)
    #eval_dict['auc'] = auc(label, output)
    return eval_dict


def split_patch(ori_img, size):
    device = ori_img.device
    b, _, h, w, d = ori_img.shape
    num_d = d // size
    num_w = w // size 
    num_h = h // size
    num_patch = num_d*num_w*num_h
    patches = torch.zeros([b*num_patch, size, size, size], dtype=torch.float).to(device)
    k = 0
    for idx in range(b):
        for d in range(num_d):
            for w in range(num_w):
                for h in range(num_h): 
                    img_patch = ori_img[idx, :, size*h: size*(h+1), size*w: size*(w+1), size*d: size*(d+1)]
                    patches[k,:] = img_patch
                    k += 1
    patches = patches.unsqueeze(dim=1)
    print("patches:", patches.shape)
    return patches, num_patch


def patch_to_bag(z_m, bag_size):
    zm_list = []
    b, dim = z_m.shape
    num = b // bag_size
    for idx in range(num):
        bag = z_m[idx*bag_size:(idx+1)*bag_size, :]
        bag = bag.unsqueeze(dim=0)
        zm_list.append(bag)
    zm_list = torch.concat(zm_list, dim=0)
    return zm_list


def random_crop_3d(volume, crop_size=(64, 64, 64)):
    # Get the dimensions of the input volume and the crop size
    volume_shape = volume.shape
    crop_height, crop_width, crop_depth = crop_size
    
    # Calculate the valid ranges for cropping along each axis
    max_crop_height = volume_shape[1] - crop_height
    max_crop_width = volume_shape[2] - crop_width
    max_crop_depth = volume_shape[3] - crop_depth
    
    # Generate random starting points along each axis
    start_height = torch.randint(0, max_crop_height + 1, (1,)).item()
    start_width = torch.randint(0, max_crop_width + 1, (1,)).item()
    start_depth = torch.randint(0, max_crop_depth + 1, (1,)).item()
    # print("start_height, start_width, start_depth:", start_height, start_width, start_depth)
    
    # Crop the sub-volume using the starting points and crop size
    cropped_sub_volume = volume[:, start_height:start_height + crop_height,
                                start_width:start_width + crop_width,
                                start_depth:start_depth + crop_depth]
    
    return cropped_sub_volume


def test_rd_patch(img, crop_size, num_for_test):
    # img.shape = (batch_size, h, w, d)
    # bag_label = label.repeat(num_for_test)
    # bag_label = torch.cat(bag_label, dim=0)
    # print("img:", img.shape)
    img = img.squeeze(dim=1)
    bag = []
    for num in range(num_for_test):
        patch = random_crop_3d(img, crop_size)
        bag.append(patch)
    bag = torch.cat(bag, dim=0)
    bag = bag.unsqueeze(dim=1)
    # print("bag:", bag.shape)
    return bag




class OriginalOrderSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.original_indices = list(range(len(data_source)))

    def __iter__(self):
        return iter(self.original_indices)

    def __len__(self):
        return len(self.data_source)

def gain_order(dataset, batch_size):
    # Create the OriginalOrderSampler
    original_order_sampler = OriginalOrderSampler(dataset)

    # Create the DataLoader with the OriginalOrderSampler
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=original_order_sampler)

    # Iterate over the batches and track the original order
    original_order = []
    for batch_x, batch_y in dataloader:
        original_order.extend(original_order_sampler.original_indices[original_order_sampler.batch_start:original_order_sampler.batch_end])

    # original_order will contain the indices of the samples in the original order






if __name__ == '__main__':
    label = [i for i in range(32)]
    dataset = TensorDataset(torch.Tensor(label))
    batch_size = 8
    index_mapping = np.arange(len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for _, batch in enumerate(dataloader):
        print("batch:", batch)
        
    for _, batch_indices in enumerate(dataloader.batch_sampler):
        for i, index in enumerate(batch_indices):
            index_mapping[batch_indices[i]] = index

    reduced_dataset = Subset(dataset, index_mapping)
    aloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for _, batch in enumerate(aloader):
        print("batch:", batch)



