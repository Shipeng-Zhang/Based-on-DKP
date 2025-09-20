import copy

import torch.nn as nn
import torchvision.models as models

import torchvision
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.p) + ', ' \
            + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)


class Normalize(nn.Module):
    def __init__(self, power=2, dim=1):
        super(Normalize, self).__init__()
        self.power = power
        self.dim = dim

    def forward(self, x):
        norm = x.pow(self.power).sum(self.dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-4)
        return out


class ResNetSimCLR(nn.Module):
    def __init__(self, base_model='resnet50', out_dim=2048, n_sampling=2, pool_len=8, normal_feature=True,
                 num_classes=500, uncertainty=False): # num_classes 分类的类别数量 # uncertainty 是否使用不确定性 # n_sampling 不确定性采样的数量 
        super(ResNetSimCLR, self).__init__()   
       
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim), 
                            "resnet50": models.resnet50(pretrained=True)}
        self.resnet = self._get_basemodel(base_model) # 选择基础模型
        self.base = nn.Sequential(*list(self.resnet.children())[:-3]) # 去掉最后的三层，保留到layer3
        dim_mlp = self.resnet.fc.in_features//2 # 2048//2=1024
        self.linear_mean =nn.Linear(dim_mlp, out_dim) # 全连接层，将1024维映射到2048维
        self.linear_var = nn.Linear(dim_mlp, out_dim) # 全连接层，将1024维映射到2048维
        self.pool_len = 8 # 特征图的宽度和高度
        self.conv_var =  nn.Conv2d(dim_mlp, dim_mlp, kernel_size=(pool_len,pool_len),bias=False) # 卷积层，感受野为8x8

        self.n_sampling = n_sampling # 不确定性采样的数量
        self.n_samples = torch.Size(np.array([n_sampling, ])) # 转换为torch.Size类型
        self.pooling_layer = GeneralizedMeanPoolingP(3) # GeM池化层，p=3

        self.l2norm_mean, self.l2norm_var, self.l2norm_sample = Normalize(2, 1), Normalize(2, 1), Normalize(2, 2) # L2归一化层 

        print('using resnet50 as a backbone')
        '''xkl add'''
        print("##########normalize matchiing feature:", normal_feature)
        self.normal_feature = normal_feature # 是否对采样的特征进行L2归一化
        self.uncertainty = uncertainty # 是否使用不确定性

        self.bottleneck = nn.BatchNorm2d(out_dim) # 批归一化层
        self.bottleneck.bias.requires_grad_(False) # bias不更新
        nn.init.constant_(self.bottleneck.weight, 1) # weight初始化为1
        nn.init.constant_(self.bottleneck.bias, 0) # bias初始化为0

        self.classifier = nn.Linear(out_dim, num_classes, bias=False) # 分类器，全连接层
        nn.init.normal_(self.classifier.weight, std=0.001) # 分类器权重初始化
        self.relu = nn.ReLU()  # ReLU激活函数

    def _get_basemodel(self, model_name):
        model = self.resnet_dict[model_name] # 获取基础模型
        return model    

    def forward(self, x, training_phase=None, fkd=False):
        BS = x.size(0) # 取出batch size
        
        out = self.base(x)  # 用基础模型提取特征，输出特征图大小为(BS, 1024, 8, 4) (batch_size, channels, height, width)
        out_mean = self.pooling_layer(out)  # global pooling (BS, 1024, 1, 1)
        out_mean = out_mean.view(out_mean.size(0), -1)  # B x 1024 # 展平为(BS, 1024)
        out_mean = self.linear_mean(out_mean)  # Bx2048 # 全连接层映射为(BS, 2048)
        # out_mean = self.l2norm_mean(out_mean)  # L2norm

        out_var = self.conv_var(out)  # conv layer (BS, 1024, 1, 1)
        out_var = self.pooling_layer(out_var)  # pooling (BS, 1024, 1, 1)
        out_var += 1e-4 
        out_var = out_var.view(out_var.size(0), -1)  # Bx1024 # 展平为(BS, 1024)
        out_var = self.linear_var(out_var)  # Bx2049 # 全连接层映射为(BS, 2048)

        out_mean=self.l2norm_mean(out_mean) # L2norm
        
        var_choice = 'L2'
        if var_choice == 'L2':
            out_var = self.l2norm_var(out_var) # L2norm
            out_var = self.relu(out_var)+ 1e-4 # ReLU激活函数，确保方差为正
        elif var_choice == 'softmax':
            out_var = F.softmax(out_var, dim=1)# Bx2049
            out_var = out_var.clone()  # gradient computation error would occur without this line
        elif var_choice=='log':
            out_var=torch.exp(0.5 * out_var)
           
        if self.uncertainty:
            BS,D=out_mean.size()   # 取出batch size和特征维度              
            tdist = MultivariateNormal(loc=out_mean, scale_tril=torch.diag_embed(out_var)) # 构建多元正态分布，均值为out_mean，协方差矩阵为对角矩阵，元素为out_var
            samples = tdist.rsample(self.n_samples)  # (n_samples, batch_size, out_dim) # 多次采样

            # if self.normal_feature:
            samples = self.l2norm_sample(samples) # L2norm

            merge_feat = torch.cat((out_mean.unsqueeze(0), samples), dim=0)  # (n_samples+1,batchsize, out_dim) # 将均值和采样的特征拼接在一起
            merge_feat = merge_feat.resize(merge_feat.size(0) * merge_feat.size(1),
                                           merge_feat.size(-1))  # ((n_samples+1)*batchsize, out_dim) # 展平为((n_samples+1)*batchsize, out_dim)
            bn_feat = self.bottleneck(
                merge_feat.unsqueeze(-1).unsqueeze(-1))  # [(n_samples+1)*batchsize, out_dim, 1, 1]
            cls_outputs = self.classifier(bn_feat[..., 0, 0])  # [(n_samples+1)*batchsize, num_classes] # 分类器输出

            merge_feat = merge_feat.resize(self.n_sampling + 1, BS,
                                           merge_feat.size(-1))  # (n_samples+1,batchsize, out_dim)
            cls_outputs = cls_outputs.resize(self.n_sampling + 1, BS,
                                             cls_outputs.size(-1))  # (n_samples+1,batchsize, num_classes)
        else:
            bn_feat = self.bottleneck(out_mean.unsqueeze(-1).unsqueeze(-1))  # [batch_size, 2048, 1, 1]
            cls_outputs = self.classifier(bn_feat[..., 0, 0])  # [batch_size, num_classes]
            cls_outputs = cls_outputs.unsqueeze(0)  # [1, batch_size, num_classes]
            merge_feat = out_mean.unsqueeze(0)  # [1, batch_size, 2048]
        if fkd:  # return all features
            return out_mean, merge_feat.permute(1, 0, 2), cls_outputs.permute(1, 0, 2), out_var, out
        if self.training:# return all features
            return out_mean, merge_feat.permute(1, 0, 2), cls_outputs.permute(1, 0, 2), out_var, out
        else:  # return mean for evalutaion
            return out_mean[:BS]


if __name__ == '__main__':
    m = ResNetSimCLR(uncertainty=True)
    m(torch.zeros(10, 3, 256, 128))
