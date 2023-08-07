import argparse
import logging
import os
import numpy as np
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchio as tio

from torch.nn.parallel import DistributedDataParallel as DDP

import torch.nn.functional as F
from dataset import ADNIdataset, OASISdataset
from dataset import ADNIdataset_sp # split train and test
from model import GFNet, UNet3d
from utils import split_dataset, calc_loss, embedding_evaluation, AverageMeter, setup_seed, calc_eval


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run(args):

    path = args.result_path + '/SL_{}'.format(args.date)
    if not os.path.exists(path):
        os.mkdir(path)
        print("make the dir")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',  filename=path + '/%s.log'%(args.date), filemode='a')
    #logging_dirc = '/.....'log—dir
    #parser.add_argument('--', type=str, ....)
    #nohup python3 main.py --log-dir='/...'
    #log的文件夹目录
    logging.info(args)
    setup_seed(args.seed)
#============================================ dataset ========================================#
    '''
        # Adding noise
        torchio.transforms.RandomNoise(mean: Union[float, Tuple[float, float]] = 0, std: Union[float, Tuple[float, float]] = (0, 0.25), **kwargs)
        # Random rotation
        transform = tio.RandomAffine(
        scales=(0.9, 1.2),
        # degree --> random rotate
        degrees=15,
        # center -- If 'image', rotations and scaling will be performed around the image center. If 'origin', rotations and scaling will be performed around the origin in world coordinates.
        )
        # Intensity adjustment/ normalization
        tio.RescaleIntensity( percentiles=(0.5, 99.5)
            out_min_max=(-1, 1), in_min_max=(ct_air, ct_bone))
        # Contrast stretching
        ### Randomly change contrast of an image by raising its values to the power gamma
        tio.RandomGamma(log_gamma=(-0.3, 0.3))
    '''
    #################################### using original data, no downsample ####################################
    '''
    # the transform sample from joint distribution, that is, each transform conduct six kinds of transformation 
    # however, the original dataset are omitted.
    transform = tio.Compose([
                tio.RandomAffine(
                    degrees=(-30,30,-30,30,-30,30),
                    center='image',  # rotations and scaling will be performed around the image center. If 'origin', rotations and scaling will be performed around the origin in world coordinates.
                    ),
                tio.OneOf({
                    tio.RandomFlip(axes=('L', )),
                    tio.RandomFlip(axes=('R', )),  
                    tio.RandomFlip(axes=('P', )),               
                }, p=0.5),
                tio.transforms.RandomNoise((0, 0.25)),
                tio.RandomElasticDeformation(p=0.5),
                tio.RandomGamma(log_gamma=(-0.3, 0.3)),
                # swap是mask掉，所以要等所有augmentation做完再加
                tio.RandomSwap(patch_size = (16,16,16), num_iterations=100, p=0.5)
                ], p = 0.5)
    '''
    transform = tio.Compose([
        tio.OneOf({
            tio.RandomAffine(
                degrees=(-30,30,-30,30,-30,30),
                center='image',  # rotations and scaling will be performed around the image center. If 'origin', rotations and scaling will be performed around the origin in world coordinates.
                ),
            tio.OneOf({
                tio.RandomFlip(axes=('L', )),
                tio.RandomFlip(axes=('R', )),  
                tio.RandomFlip(axes=('P', )),               
            }),
            tio.transforms.RandomNoise((0, 0.25)),
            tio.RandomElasticDeformation(p=0.5),
            tio.RandomGamma(log_gamma=(-0.3, 0.3)),
            # swap是mask掉，所以要等所有augmentation做完再加
            tio.RandomSwap(patch_size = (16,16,16), num_iterations=100)})
        ], p = args.prob)  # only half the probability to transform original pic
    

    if args.name == 'ADNI1':
        train_dataset = ADNIdataset(args.label_path, args.img_path, 'train', transform=transform)
        test_dataset = ADNIdataset(args.label_path, args.img_path, 'test', transform=None)
    else:
        train_dataset = OASISdataset(args.label_path, args.img_path, 'train', transform=transform)
        test_dataset = OASISdataset(args.label_path, args.img_path, 'test', transform=None)    
    print('len(train_dataset):', len(train_dataset))
    print("any intersection:", set(train_dataset.patients_list)&set(test_dataset.patients_list), len(train_dataset.patients_list), len(test_dataset.patients_list))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    '''
    dataset = ADNIdataset(args.label_path, args.img_path, transform=transform)
    train_indice, test_indice = split_dataset(args, dataset)
    train_sampler = SubsetRandomSampler(train_indice)
    test_sampler = SubsetRandomSampler(test_indice)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=4)
    '''
#========================================== model & optimizer ========================================#
    # gfnet 只要两层 learning rate = 
    model = GFNet(depth=args.gf_depth, num_classes=args.gfopc, in_channels=args.out_channel)
    # model = GFNet_ds(img_size=[96, 112, 96], patch_size=[16,16,16], embed_dim=4096, feature_dim=args.bag_dim)
    print("model:", model)
    print('#latent_encoder parameters:', sum(param.numel() for param in model.parameters())) 
   
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model).to(device)

    if args.optim == 'sgd':
        opt_l = torch.optim.SGD(model.parameters(), lr=args.l_lr, momentum=0.9, weight_decay=1e-5)
    else:
        opt_l = torch.optim.Adam(model.parameters(), lr=args.l_lr, weight_decay=1e-5)


    criterion = nn.CrossEntropyLoss()
    max_val_acc = 0
    best_epoch = 0
#========================================= load latent vectors =======================================
    for epoch in range(args.epochs):

        losses = AverageMeter()
        acc_s = AverageMeter()
        f1_s = AverageMeter()
        training_process = tqdm(train_loader, desc='training')
        # training_process = tqdm(train_loader, desc='training')
        model.train()
        for idx, batch in enumerate(training_process):
            _, img, label = batch
            label = label.to(device)
            img = img.to(device) 
            # opt_l.zero_grad()
            model.zero_grad()
            y_pred = model(img)
            #y_pred = torch.argmax(y_pred, dim=1)
            loss = criterion(y_pred, label)
            dict = calc_eval(y_pred, label)
            acc_s.update(dict['acc'], img.size(0))
            f1_s.update(dict['f1'], img.size(0))
            loss.backward()
            losses.update(loss.item(), img.size(0))
            opt_l.step()

            # logging.info("acc: {} f1: {} spe: {} sen: {}".format(dict['acc'], dict['f1'], dict['spe'], dict['sen']))
        logging.info("train: acc: {} f1: {} ".format(acc_s.avg, f1_s.avg))

        logging.info(
                'Epoch {}, loss: {}'.format(epoch + 1, losses.avg))

        if epoch % 30 == 0:
            state = {'latent_encoder': model.state_dict(),
                    'opt_l': opt_l.state_dict()}
            torch.save(state, path + '/b{}_{}_{}.pth'.format(args.batch_size, epoch, args.reg))

        e_acc_s = AverageMeter()
        e_f1_s = AverageMeter()
        model.eval()
        testing_process = tqdm(test_loader, desc='testing')
        with torch.no_grad():
            for idx, batch in enumerate(testing_process):
                _, img, label = batch
                img = img.to(device)
                label = label.to(device)  
                output = model(img)
                eval_dict = calc_eval(output, label)
                e_acc_s.update(eval_dict['acc'], img.size(0))
                e_f1_s.update(eval_dict['f1'], img.size(0))
                # 附上auc** area under curve: value; ROC: xxx curve

            logging.info("val: acc: {} f1: {}".format(e_acc_s.avg, e_f1_s.avg))
        
        if max_val_acc <= e_acc_s.avg:
            max_val_acc = e_acc_s.avg
            best_epoch = epoch

    logging.info(
        'best epoch: {}, max val acc: {}'. format(best_epoch, max_val_acc)
    )

    state = {'latent_encoder': model.state_dict(),
            'opt_l': opt_l.state_dict()}
    torch.save(state, path + '/best_latent_encoder' + '_%s.pth'%args.reg)


def arg_parse():
    parser = argparse.ArgumentParser(description='ADNI classification')
    parser.add_argument('--name', type=str, default='ADNI1',
                        help='name of dataset')
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed')
    parser.add_argument('--label_path', type=str, default='/home/zhang_istbi/zhangsj/ACGF/ADNI.csv',
                        help='label path')
    parser.add_argument('--img_path', type=str, default='/home/zhang_istbi/zhangsj/ACGF/processed_ADNI',
                        help='data path')
    parser.add_argument('--pretrain_path', type=str, default='/home/zhang_istbi/zhangsj/ACGF/result/best_latent_encoder_True.pth',
                        help='data path')
    parser.add_argument('--pretrain', type=bool, default=False,
                        help='pretrain or not')
    parser.add_argument('--size', type=tuple, default=(32,32,32),
                        help='size of data')
    parser.add_argument('--shuffle_dataset', type=str,
                    default='True', help='shuffle indice')
    parser.add_argument('--l_lr', type=float, default=0.0005,
                        help='latent encoder Learning rate.')
    parser.add_argument('--prob', type=float, default=1.0,
                        help='augmentation probability.')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size')
    parser.add_argument('--reg', type=str, default='True',
                        help='regularization')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Train Epochs')
    parser.add_argument('--out_channel', type=int, default=1,
                        help='output channel of unet3d')
    parser.add_argument('--gf_depth', type=int, default=4,
                        help='depth of gfnet')
    # sgd容易陷入local minimum
    parser.add_argument('--optim', type=str, default='Adam',
                        help='type of optimizer')
    parser.add_argument('--result_path', type=str, default='/home/zhang_istbi/zhangsj/ACGF/result',
                        help='path to save')
    parser.add_argument('--date', type=str, default='731',
                        help='date and num ')   
    parser.add_argument('--gfopc', type=int, default=2,
                        help='output channels of gfnet')
                        
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    run(args)