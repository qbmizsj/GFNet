from utils import read_json, data_split
from model_wrapper_gf import GFNet_Wrapper
import torch
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import torch
torch.backends.cudnn.benchmark = True


def gfnet_main(seed):
    gfnet_setting = config['gfnet']
    for exp_idx in range(repe_time):
        cnn = GFNet_Wrapper(
                          batch_size=gfnet_setting['batch_size'],
                          balanced=gfnet_setting['balanced'],
                          Data_dir=gfnet_setting['Data_dir'],
                          exp_idx=exp_idx,
                          seed=seed,
                          model_name='GFNet',
                          metric='accuracy')
        cnn.train(lr=gfnet_setting['learning_rate'],
                  epochs=gfnet_setting['train_epochs'])
        cnn.test()

if __name__ == '__main__':
    config = read_json('./config.json')
    seed , repe_time = 1000, config['repeat_time']
    data_split(repe_time = repe_time)
    with torch.cuda.device(0):
        gfnet_main(seed)
