from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image

class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset) # 返回数据集的大小

    def __getitem__(self, indices):
            return self._get_single_item(indices) 

    def _get_single_item(self, index):
        # print(self.dataset)
        try:
            fname, pid, camid, domain = self.dataset[index] # dataset中的每个元素包含 (img_path, pid, camid, domain-id)
        except:
            fname, pid, camid = self.dataset[index]
        fpath = fname # 图像路径
        if self.root is not None:
            fpath = osp.join(self.root, fname) # 如果提供了根目录，则将其与图像路径连接起来

        img = Image.open(fpath).convert('RGB') # 打开图像并转换为RGB格式

        if self.transform is not None:
            img = self.transform(img) # 对图像进行预处理

        return img, fname, pid, camid # 返回图像张量, 图像路径, 行人ID, 相机ID
