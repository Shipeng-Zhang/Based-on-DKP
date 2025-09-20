from __future__ import absolute_import
from collections import defaultdict
import math

import numpy as np
import copy
import random
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples).tolist()
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)

class MultiDomainRandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances

        self.domain2pids = defaultdict(list)
        self.pid2index = defaultdict(list)

        for index, (_, pid, _, domain) in enumerate(data_source):
            if pid not in self.domain2pids[domain]:
                self.domain2pids[domain].append(pid)
            self.pid2index[pid].append(index)

        self.pids = list(self.pid2index.keys())
        self.domains = list(sorted(self.domain2pids.keys()))

        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):

        ret = []
        domain2pids = copy.deepcopy(self.domain2pids)
        for _ in range(8):
            for domain in self.domains:
                pids = np.random.choice(domain2pids[domain], size=8, replace=False)
                for pid in pids:
                    idxs = copy.deepcopy(self.pid2index[pid])
                    idxs = np.random.choice(idxs, size=2, replace=False)
                    ret.extend(idxs)
        return iter(ret)

class RandomMultipleGallerySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source # data_source (path, pid, camid, domain-id)
        self.index_pid = defaultdict(int) #index 2 pid 字典
        self.pid_cam = defaultdict(list) # pid 2 camid 列表
        self.pid_index = defaultdict(list) # pid 2 index 列表
        self.num_instances = num_instances # 每个batch中每个ID的样本数量

        try :
            for index, (_, pid, cam) in enumerate(data_source):
                self.index_pid[index] = pid # index 2 pid
                self.pid_cam[pid].append(cam) # pid 2 camid
                self.pid_index[pid].append(index) #index 2 index
        except:
            for index, (_, pid, cam,_,_) in enumerate(data_source):
                self.index_pid[index] = pid
                self.pid_cam[pid].append(cam)
                self.pid_index[pid].append(index)
        #print(self.pid_cam)
        self.pids = list(self.pid_index.keys()) # 所有的pid(不重复)
        self.num_samples = len(self.pids) # ID的数量，即类别的数量

    def __len__(self):
        return self.num_samples * self.num_instances # 每个ID的样本数量乘以ID的数量

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist() # 将所有的ID随机打乱，方便随机选取样本
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]]) # 随机选择一个样本在data_source中的索引

            _, i_pid, i_cam = self.data_source[i] # 通过索引获取该样本的 pid 和 camid
            #_, i_pid, i_cam = self.data_source[i]

            ret.append(i) # 将样本的索引加入结果列表

            pid_i = self.index_pid[i] # 获取该样本的 pid
            cams = self.pid_cam[pid_i] # 获取该 pid 对应的所有 camid 列表
            index = self.pid_index[pid_i] # 获取该 pid 对应的所有样本索引列表
            select_cams = No_index(cams, i_cam) # 获取除当前样本 camid 外的其他 camid 列表

            if select_cams:

                if len(select_cams) >= self.num_instances: # 相机数量大于等于ID需要的样本数量
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=False) # 不放回抽样
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=True) # 放回抽样

                for kk in cam_indexes:
                    ret.append(index[kk]) # 将样本索引加入结果列表

            else: # 该ID的样本都来自同一相机
                select_indexes = No_index(index, i) # 除当前样本外的其他样本索引
                if (not select_indexes): continue # 该ID只有一个样本，跳过
                if len(select_indexes) >= self.num_instances: # 该ID的样本数量大于等于需要的样本数量
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False) # 不放回抽样
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True) # 放回抽样

                for kk in ind_indexes:
                    ret.append(index[kk]) # 将样本索引加入结果列表


        return iter(ret) # 返回结果(索引)的迭代器
