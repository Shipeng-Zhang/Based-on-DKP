import torchvision.transforms as T
import copy
import os.path
import os
from reid.utils.feature_tools import *
import lreid_dataset.datasets as datasets
from reid.utils.data.sampler import RandomMultipleGallerySampler
from reid.utils.data import IterLoader
import numpy as np


def get_data(data_dir,height, width, batch_size, workers,num_instances,cam_filter = None,name ='market1501',select_num=0):
    # root = osp.join(data_dir, name)
    root = data_dir

    if cam_filter is not None:
        cam_id = int(cam_filter.split("cam")[-1])
        cam_filter = cam_id
    
    
    dataset = datasets.create(name,root,cam_filter) # 创建数据集实例，datasets.train, datasets.query, datasets.gallery
    num_classes = dataset.num_classes

    # '''select some persons for training'''
    # if select_num > 0: # 限制训练集中的ID数量
    #     train = []
    #     for instance in dataset.train:
    #         if instance[1] < select_num: # 代表选择前 select_num 个ID进行训练
    #             # new_id=id_2_id[instance[1]]
    #             train.append((instance[0], instance[1], instance[2], instance[3]))  #img_path, pid, camid, domain-id

        # dataset.train = train # 更新训练集
    # dataset.num_train_pids = select_num # 更新训练集中的ID数量
    dataset.num_train_imgs = len(dataset.train) # 更新训练集中的图片数量

   

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) # 图像归一化 符合ImageNet标准

    train_set = sorted(dataset.train) # 按照path,pid, camid排序

    iters = int(len(train_set) / batch_size) # 计算每个epoch的迭代次数
    # num_classes = dataset.num_train_pids # 训练集中的ID数量，即类别数

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3), # 调整图像大小 interpolation = 3 ：双三次插值
        T.RandomHorizontalFlip(p=0.5), # 50%概率水平翻转
        T.Pad(10), # 四周填充10个像素
        T.RandomCrop((height, width)), # 随机裁剪到固定大小
        T.ToTensor(), # 转换为张量
        normalizer, # 归一化
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]) # 随机遮挡 50%概率遮挡，遮挡区域像素值为均值
    ]) # 对训练集的组合变换，增强数据

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3), # 调整图像大小 interpolation = 3 ：双三次插值
        T.ToTensor(), # 转换为张量
        normalizer # 归一化
    ]) # 对测试集的组合变换

    rmgs_flag = num_instances > 0 # num_instances 每个batch中每个ID的样本数量
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances) # 图片采样器
    else:
        sampler = None


    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir,transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters) # 训练集数据加载器

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer), 
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True) # 测试集数据加载器

    init_loader = DataLoader(Preprocessor(train_set, root=dataset.images_dir,transform=test_transformer),
                             batch_size=128, num_workers=workers,shuffle=False, pin_memory=True, drop_last=False) # 用于初始化的训练集数据加载器
    if cam_filter is not None:
        return [dataset, num_classes, train_loader, test_loader, init_loader, name+'_cam'+str(cam_filter)] # 返回数据集实例，类别数，训练集数据加载器，测试集数据加载器，用于初始化的训练集数据加载器，数据集名称
    else:
        return [dataset, num_classes, train_loader, test_loader, init_loader, name]

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) # 图像归一化 符合ImageNet标准

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3), # 调整图像大小 interpolation = 3 ：双三次插值
        T.ToTensor(), # 转换为张量
        normalizer # 归一化
    ]) # 对测试集的组合变换

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery)) # 测试集由query和gallery组成

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True) # 测试集数据加载器

    return test_loader 


# 创建训练集和测试集的数据加载器
def build_data_loaders(cfg, training_set, testing_only_set, toy_num=0):
    # Create data loaders
    data_dir = cfg.data_dir # 数据集目录
    height, width = (256, 128) # 图像输入大小
    training_loaders = [get_data(data_dir, height, width, cfg.batch_size, cfg.workers,
                                 cfg.num_instances, cam_filter,name = 'market1501',select_num=0) for cam_filter in training_set] # 训练集数据加载器 返回值training_loaders包括 dataset, num_classes, train_loader, test_loader, init_loader, name

  
    testing_loaders = [get_data(data_dir, height, width, cfg.batch_size, cfg.workers,
                            cfg.num_instances)] # 测试集数据加载器 # 返回值testing_loaders 包括 dataset, num_classes, train_loader, test_loader, init_loader, name
    return training_loaders, testing_loaders # 返回训练集和测试集的数据加载器
