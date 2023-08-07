import torch
from torch import dtype, float32
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

from torch.nn.modules.module import Module
import logging
import math
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # add one more activation function
        x = self.act(x)
        x = self.drop(x)
        return x


'''
class Global_Filter(nn.Module):
    def __init__(self, dim, h, w, d ):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h,w,d,dim,2, dtype=torch.float32)*0.02)
        self.w = w
        self.h = h
        self.d = d
    def forward(self, x, spatial_size = None):
'''
#dim = 1000


class Global_Filter(nn.Module):
    def __init__(self, p, h=9, w=10, d=5, dim=256):
        super().__init__()
        # 生成可学习权重的做法
        # 通过rand初始生成一个指定尺寸的随机tensor，再通过nn.Parameter实现梯度下降
        # d变为w的一半，向下取整+1。这是快速傅立叶变换的要求
        # 这里使用了广播， 该机制是倒着看
        # 快速傅立叶变换，无论是在实数域还是在复数域，都要压缩为原来的一半
        # 所以complex_weight是频域上的learnable tensor
        d = d//2+1
        self.complex_weight = nn.Parameter(torch.randn(
            h, w, d, dim, 2, dtype=torch.float32) * 1.5)
        self.patch = (p,p,p)
        self.h = h
        self.w = w
        self.d = d
        self.dim = dim

    def forward(self, x):
        # N dim c =B 64 36
        # N c dim
        # B 4,4,4,36
        # torch.Size([36, 32768, 256])
        B, N, C = x.shape
        x = x.to(torch.float32)
        # x = x.view(B, 18, 21, 18, 1000)
        h,w,d=self.patch
        x = x.view(B, h,w,d, self.dim)
        # 实数的傅立叶变换
        x = torch.fft.rfftn(x, dim=(1, 2, 3), norm='ortho')
        # 复数的可学习权重
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        # x = torch.fft.irfftn(x, s=(18, 21, 18), dim=(1, 2, 3), norm='ortho')
        x = torch.fft.irfftn(x, s=self.patch, dim=(1, 2, 3), norm='ortho')
        x = x.reshape(B, N, C)
        return x

# Block [N x 3888/8 x 1000] ----> [N x 3888 x 1000]
class tiny_GF(nn.Module):
    def __init__(self, patch_size, img_size, in_channel, dim, num_classes, norm_layer=nn.LayerNorm):
        super().__init__()
        h, w, d = img_size
        self.norm1 = norm_layer(dim)
        self.filter = Global_Filter(p=patch_size, dim=dim, h=h, w=w, d=d)
        self.norm2 = norm_layer(dim)
        self.head = nn.Sequential(
            nn.Linear(in_channel, 16),
            nn.GELU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = x + self.norm2(self.filter(self.norm1(x)))
        # b, 64, 36 --> b, 64
        x = x.mean(2)
        x = self.head(x)
        return x
    
    


class Block(nn.Module):
    def __init__(self, p=32, dim=256, mlp_ratio=2., drop=0.5, drop_path=0.6, act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=18, w=21, d=10):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = Global_Filter(p=p, dim=dim, h=h, w=w, d=d)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + \
            self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x


class PatchEmbed(nn.Module):
    #image to patch embedding
    def __init__(self, in_channels, out_channel):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
       
        N, C, H, W, D = x.size()
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class GFNet_p(nn.Module):
    # 181//2+1, 217//2+1, 181//2+1
    def __init__(self, img_size=(181//2+1, 217//2+1, 181//2+1), embed_dim=256, feature_dim=64, num_classes=16, in_channels=1, drop_rate=0.5, depth=1, mlp_ratio=2., representation_size=None, uniform_drop=False, drop_path_rate=0.6, norm_layer=False, dropcls=0.25):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        h, w, d = img_size
        self.pos_embed = nn.Parameter(torch.zeros(1, h*w*d, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.patch_embed = PatchEmbed(in_channels=in_channels, out_channel=embed_dim)
        
        if uniform_drop:
            print('using uniform droppath with expected rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]
        else:
            print('using linear droppath with expected rate', drop_path_rate * 0.5)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(p=h, dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate,
                  drop_path=dpr[i], norm_layer=norm_layer, h=h, w=w, d=d)
            for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)

        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        self.embedding = nn.Linear(self.embed_dim, feature_dim)

        # tensor clip
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.Linear(64, num_classes)
        )


    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x) 
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x).mean(1)
        ### add embedding ###
        return x

    def forward(self, x, feature_only = False): 
        features = self.forward_features(x)
        x = self.embedding(features)
        if feature_only:
            return features, x


if __name__ == '__main__':
    x = torch.randn(2, 1, 16,16,16)
    # xp = x[:,:,::2,::2,::2]
    # print("xp:", xp.shape)
    net = GFNet(img_size=(16,16,16))
    y = net(x)
    print(y.shape)