from __future__ import absolute_import
from collections import OrderedDict
import torch

from ..utils import to_torch

def extract_cnn_feature(model, inputs,training_phase=None):
    model.eval() # 模型不参与训练
    with torch.no_grad(): # 不计算梯度
        inputs = to_torch(inputs).cuda() # 转换为张量并放到GPU上


        Expand=False 
        if inputs.size(0)<2:  # 如果batch size小于2，进行扩展，避免BN层出错
            Pad=inputs[:1]  
            inputs=torch.cat((inputs,Pad),dim=0)  # 复制第一张图片扩展batch size
            Expand=True 


        outputs = model(inputs,training_phase=training_phase) # 前向传播
        outputs = outputs.data.cpu() # 转换为CPU张量

        if Expand:
            outputs=outputs[:-1] # 去掉扩展的那张图片

        return outputs

