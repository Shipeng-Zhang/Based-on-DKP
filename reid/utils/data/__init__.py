from __future__ import absolute_import

from .base_dataset import BaseImageDataset
from .preprocessor import Preprocessor
from .dataset import Dataset

class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if (self.length is not None):
            return self.length
        return len(self.loader)

    def new_epoch(self): # 重置迭代器
        self.iter = iter(self.loader)

    def next(self): # 获取下一个batch的数据
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)