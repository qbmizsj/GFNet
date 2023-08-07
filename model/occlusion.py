import sys
import time
import torch
import os
import torch.nn as nn
import numpy as np
# from utils import EncoderBlock, DecoderBlock
from .utils import EncoderBlock, DecoderBlock
sys.path.append("..")
import gc

gc.collect()
torch.cuda.empty_cache()



#os.environ['CUDA_VISIBLE_DEVICES']=" 1"

class UNet3d(nn.Module):

    def __init__(self, in_channels, out_channels, model_depth=4, final_activation="sigmoid"):
        super(UNet3d, self).__init__()
        self.encoder = EncoderBlock(in_channels=in_channels, model_depth=model_depth)
        self.decoder = DecoderBlock(out_channels=out_channels, model_depth=model_depth)
        #self.f =  nn.ConstantPad3d((5,6,3,4,5,6),0)
        if final_activation == "sigmoid":
            self.sigmoid = nn.Sigmoid()
        else:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        _, _, _, d, _ = x.shape
        device = x.device
        x, downsampling_features = self.encoder(x)
        x = self.decoder(x, downsampling_features)
        x = self.sigmoid(x)
        # print("indice:", indice.shape)
        # print('indice_low shape is:', indice_low.shape)
        # print('indice_high shape is:', indice_up.shape)
        # print('----------------------------')
        # x = x[:,:,5:-6,3:-4,5:-6]
        return x


 


if __name__ == "__main__":
    #device = torch.device("cuda:0")
    os.environ['CUDA_VISIBLE_DEVICES']="0, 1"
    inputs = torch.randn(4, 1, 61,61,61).cuda()
    f =  nn.ConstantPad3d((1,2,1,2,1,2),0)
    inputs = f(inputs)
    print("The shape of inputs: ", inputs.shape)
    model = UNet3d(in_channels=1, out_channels=1).cuda()
    #如果需要用到的话
    inputs = inputs.cuda()
    # model = torch.nn.DataParallel(model).cuda()
    x = model(inputs)
    print('The output dimsion of UNet3D is:',x.size())
    #print(model)
