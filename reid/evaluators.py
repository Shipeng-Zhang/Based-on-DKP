from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import numpy as np
import torch

from .evaluation_metrics import cmc, mean_ap, mean_ap_cuhk03
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from torch.nn import functional as F

def extract_features(model, data_loader,training_phase=None):
    model.eval() # 模型不参与训练
    try:
        model = model.module # 适配多GPU
    except:
        model=model # 单GPU
    batch_time = AverageMeter() # 计算每个batch的时间
    data_time = AverageMeter() # 计算每个batch加载数据的时间

    features = OrderedDict() # 有序字典，存储特征
    labels = OrderedDict() # 有序字典，存储标签

    end = time.time() # 记录时间
    with torch.no_grad(): # 不计算梯度
        for i, (imgs, fnames, pids, cids) in enumerate(data_loader): # 遍历数据集
            data_time.update(time.time() - end) # 计算加载数据时间

            outputs = extract_cnn_feature(model, imgs,training_phase=training_phase) # 提取特征张量
            for fname, output, pid in zip(fnames, outputs, pids): # 遍历一个batch中的每个样本
                features[fname] = output # 存储特征
                labels[fname] = pid # 存储标签

            batch_time.update(time.time() - end) # 计算处理一个batch的时间
            end = time.time() # 更新结束时间
 
    return features, labels # 返回特征和标签

def extract_features_print(model, data_loader,training_phase=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, cids) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs,training_phase=training_phase)
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

    return features, labels



# 计算查询库和画廊库之间的欧式距离
def pairwise_distance(features, query=None, gallery=None, metric=False):
    if query is None and gallery is None:
        n = len(features) # 样本数量
        x = torch.cat(list(features.values())) # 将所有特征拼接成一个大张量
        x = x.view(n, -1) # 调整形状为 (n, feature_dim)
        if metric is not False:
            x=F.normalize(x, p=2, dim=1) # L2归一化
            # x = metric.transform(x)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2 # 计算每个样本的平方和并扩展
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t()) # 计算欧式距离矩阵
        return dist_m # 返回距离矩阵

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0) # 找到查询集中图片的特征张量
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0) # 找到画廊集中图片的特征张量
    m, n = x.size(0), y.size(0) # 查询集和画廊集的样本数量
    x = x.view(m, -1) # 调整形状为 (m, feature_dim)
    y = y.view(n, -1) # 调整形状为 (n, feature_dim)
    if metric is not False:
        x = F.normalize(x, p=2, dim=1) # L2归一化
        y = F.normalize(y, p=2, dim=1) # L2归一化
        # x = metric.transform(x)
        # y = metric.transform(y)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()  # 计算每个样本的平方和并扩展
    dist_m.addmm_(1, -2, x, y.t()) # 计算欧式距离矩阵 1：dist_m = 1*dist_m + (-2)*(x @ y^T)
    return dist_m, x.numpy(), y.numpy() # 返回距离矩阵，查询集特征，画廊集特征

def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False, cuhk03=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query] # 提取查询集的ID
        gallery_ids = [pid for _, pid, _ in gallery] # 提取画廊集的ID
        query_cams = [cam for _, _, cam in query] # 提取查询集的相机ID
        gallery_cams = [cam for _, _, cam in gallery] # 提取画廊集的相机ID
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    if cuhk03:
        mAP = mean_ap_cuhk03(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    else:
        mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams) # 返回mAP
    print('Mean AP: {:4.1%}'.format(mAP))

    if (not cmc_flag):
        return mAP
    '''
    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),
        'cuhk03': dict(separate_camera_set=True,
                           single_gallery_shot=True,
                           first_match_break=False)
                }
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}
    '''
    if cuhk03:
        cmc_configs = {
        'cuhk03': dict(separate_camera_set=True,
                           single_gallery_shot=True,
                           first_match_break=False)
                }
        
        cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}
        print('CUHK03 CMC Scores:')
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}'
                .format(k,
                        cmc_scores['cuhk03'][k-1]))
        return cmc_scores['cuhk03'][0], mAP
    
    else:
        cmc_configs = {
            'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),
                }
        cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}
        print('CMC Scores:')
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}'
                .format(k,
                        cmc_scores['market1501'][k-1]))
        return cmc_scores['market1501'][0], mAP

class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, metric=None, cmc_flag=False,
                 rerank=False, pre_features=None, cuhk03=False,training_phase=None):
        if (pre_features is None):
            features, _ = extract_features(self.model, data_loader,training_phase=training_phase) # 返回两个字典 features, labels
        else:
            features = pre_features


        # 返回距离矩阵 查询集的特征(numpy) 画廊集的特征(numpy)
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery, metric=metric) 
        # 返回mAP 和CMC
        results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag, cuhk03=cuhk03)
        if (not rerank):
            return results

        print('Applying person re-ranking ...')
        distmat_qq = pairwise_distance(features, query, query, metric=metric)
        distmat_gg = pairwise_distance(features, gallery, gallery, metric=metric)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)

