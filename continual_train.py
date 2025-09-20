from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import sys

from torch.backends import cudnn
import torch.nn as nn
import random
from reid.evaluators import Evaluator
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.utils.lr_scheduler import WarmupMultiStepLR
from reid.utils.feature_tools import *
from reid.models.layers import DataParallel
from reid.models.resnet_uncertainty import ResNetSimCLR
from reid.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter

from lreid_dataset.datasets.get_data_loaders import build_data_loaders
from tools.Logger_results import Logger_res

from lreid_dataset import datasets
import time
from lreid_dataset.datasets.get_data_loaders import get_data

def main():
    args = parser.parse_args()

    if args.seed is not None: # seed:便于复现
        print("setting the seed to",args.seed)
        random.seed(args.seed)  # 
        np.random.seed(args.seed) # np的随机数种子
        torch.manual_seed(args.seed) # torch的CPU随机数种子
        torch.cuda.manual_seed(args.seed) # torch的GPU随机数种子
        torch.cuda.manual_seed_all(args.seed) # if you are using multi-GPU

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True # 保证每次结果一样

    main_worker(args)


def main_worker(args):
    log_name = 'log.txt'
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    else:
        log_dir = osp.dirname(args.test_folder)
        sys.stdout = Logger(osp.join(log_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))
    log_res_name='log_res.txt'
    logger_res=Logger_res(osp.join(args.logs_dir, log_res_name))    # record the test results
    

    """
    loading the datasets:
    setting： 1 or 2 
    """
    if 1 == args.setting:
        training_set = ['market1501', 'cuhk_sysu', 'dukemtmc', 'msmt17', 'cuhk03'] # 训练顺序
    else:
        training_set = ['dukemtmc', 'msmt17', 'market1501', 'cuhk_sysu', 'cuhk03'] # 训练顺序
    # all the revelent datasets
    all_set = ['market1501', 'dukemtmc', 'msmt17', 'cuhk_sysu', 'cuhk03',
               'cuhk01', 'cuhk02', 'grid', 'sense', 'viper', 'ilids', 'prid']  # 'sense','prid' # 所有的数据集
    # the datsets only used for testing
    testing_only_set = [x for x in all_set if x not in training_set] # 测试数据集
    # get the loders of different datasets
    # all_train_sets: [dataset, num_classes, train_loader, test_loader, init_loader, name]
    # all_test_only_sets: [dataset, num_classes, train_loader, test_loader, init_loader, name]
    all_train_sets, all_test_only_sets = build_data_loaders(args, training_set, testing_only_set)    
    
    first_train_set = all_train_sets[0] # 第一个训练集的[dataset, num_classes, train_loader, test_loader, init_loader, name]
    
    model=ResNetSimCLR(num_classes=first_train_set[1], uncertainty=True,n_sampling=args.n_sampling) 

    model.cuda() # 开始训练状态
    model = DataParallel(model)  # 多GPU训练    
    writer = SummaryWriter(log_dir=args.logs_dir) # tensorboard
    # Load from checkpoint
    '''test the models under a folder'''
    if args.test_folder: # 测试模式
        ckpt_name = [x + '_checkpoint.pth.tar' for x in training_set]   # obatin pretrained model name 获得模型名称
        checkpoint = load_checkpoint(osp.join(args.test_folder, ckpt_name[0]))  # load the first model 加载第一个模型
        copy_state_dict(checkpoint['state_dict'], model)     #  把第一个模型加载到model中  
        for step in range(len(ckpt_name) - 1):
            model_old = copy.deepcopy(model)    # backup the old model            
            checkpoint = load_checkpoint(osp.join(args.test_folder, ckpt_name[step + 1]))
            copy_state_dict(checkpoint['state_dict'], model) # 把新模型加载到model中                      
           
            model = linear_combination(args, model, model_old, 0.5) # 融合新旧模型

            save_name = '{}_checkpoint_adaptive_ema.pth.tar'.format(training_set[step+1]) # 保存模型名称
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': 0,
                'mAP': 0,
            }, True, fpath=osp.join(args.logs_dir, save_name))   # 保存模型  
        test_model(model, all_train_sets, all_test_only_sets, len(all_train_sets)-1,logger_res=logger_res)

        exit(0)
    

    # resume from a model
    if args.resume: # 恢复训练
        checkpoint = load_checkpoint(args.resume) # 加载模型
        copy_state_dict(checkpoint['state_dict'], model) # 加载到model中
        start_epoch = checkpoint['epoch'] # 开始的epoch
        best_mAP = checkpoint['mAP'] # 最佳mAP
        print("=> Start epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))
   
    # Evaluator
    if args.MODEL in ['50x']:
        out_channel = 2048
    else:
        raise AssertionError(f"the model {args.MODEL} is not supported!")


    # train on the datasets squentially
    for set_index in range(0, len(training_set)):  
        model_old = copy.deepcopy(model)
        model = train_dataset(args, all_train_sets, all_test_only_sets, set_index, model, out_channel,
                                            writer,logger_res=logger_res)
        if set_index>0:
            model = linear_combination(args, model, model_old, 0.5)
            test_model(model, all_train_sets, all_test_only_sets, set_index, logger_res=logger_res)    
    print('finished')



# 用来提取旧数据集的特征分布
def obtain_old_types( all_train_sets, set_index, model):
    dataset_old, num_classes_old, train_loader_old, _, init_loader_old, name_old = all_train_sets[
        set_index - 1]  # 旧数据集的状态
    # 旧数据集的所有特征 所有标签 文件名 相机名 均值特征 标签列表 均值方差 所有方差
    features_all_old, labels_all_old, fnames_all, camids_all, features_mean, labels_named, vars_mean,vars_all = extract_features_uncertain(model,
                                                                                                                  init_loader_old,
                                                                                                                  get_mean_feature=True)  # init_loader is original designed for classifer init
    features_all_old = torch.stack(features_all_old) # 转换为张量
    labels_all_old = torch.tensor(labels_all_old).to(features_all_old.device) # 转换为张量
    features_all_old.requires_grad = False # 不需要梯度
    return features_all_old, labels_all_old, features_mean, labels_named,vars_mean,vars_all

# 用于训练模型
def train_dataset(args, all_train_sets, all_test_only_sets, set_index, model, out_channel, writer,logger_res=None):
    if set_index>0: # 不是第一个数据集
        # 提取旧数据集的所有特征 所有标签 均值特征 标签列表 均值方差 所有方差
        features_all_old, labels_all_old,features_mean, labels_named,vars_mean,vars_all=obtain_old_types( all_train_sets,set_index,model)
        proto_type={} # 存储旧数据集的信息
        proto_type[set_index-1]={
                "features_all_old":features_all_old,
                "labels_all_old":labels_all_old,
                'mean_features':features_mean,
                'labels':labels_named,
                'mean_vars':vars_mean,
                "vars_all":vars_all
            }
    else:
        proto_type=None 
    dataset, num_classes, train_loader, test_loader, init_loader, name = all_train_sets[
        set_index]  # status of current dataset     # 获得当前训练集 类别数 训练集加载器 测试集加载器 初始化集加载器 名称

    Epochs= args.epochs0 if 0==set_index else args.epochs          

    if set_index<=1:
        add_num = 0
        old_model=None
    else:
        add_num = sum(
            [all_train_sets[i][1] for i in range(set_index - 1)])  # get person number in existing domains
    
    
    if set_index>0: # 不是第一个数据集
        '''store the old model''' 
        old_model = copy.deepcopy(model) 
        old_model = old_model.cuda()
        old_model.eval()  # 保存旧模型

        # after sampling rehearsal, recalculate the addnum(historical ID number)
        add_num = sum([all_train_sets[i][1] for i in range(set_index)])  # 保存之前所有数据集的类别数
        # Expand the dimension of classifier
        org_classifier_params = model.module.classifier.weight.data # 获得旧分类器权重参数
        model.module.classifier = nn.Linear(out_channel, add_num + num_classes, bias=False) # 新分类器，分类数为旧类别数+新类别数
        model.module.classifier.weight.data[:add_num].copy_(org_classifier_params) # 复制旧分类器前add_sum行参数到新分类器前面部分
        model.cuda()  # 重新放到GPU上
        # Initialize classifer with class centers    
        class_centers = initial_classifier(model, init_loader) # 每个类中心
        model.module.classifier.weight.data[add_num:].copy_(class_centers) # 为新类别的分类器提供合理的初始值
        model.cuda() # 重新放到GPU上

    # Re-initialize optimizer 重新初始化优化器
    params = []
    for key, value in model.named_params(model): # model.named_parameters()返回模型的所有参数
        if not value.requires_grad: # 如果不需要梯度
            print('not requires_grad:', key)
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}] # 添加参数
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params) # Adam优化器
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=args.momentum)  # SGD优化器
    # Stones=args.milestones
    Stones = [20, 30] if name == 'msmt17' else args.milestones # msmt17数据集训练时间长，调整学习率的epoch也相应变大
    lr_scheduler = WarmupMultiStepLR(optimizer, Stones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step) # 学习率调整策略
    
    
    # 创建训练类，把训练逻辑进行封装
    trainer = Trainer(args, model,  writer=writer)

    print('####### starting training on {} #######'.format(name))
    for epoch in range(0, Epochs):

        train_loader.new_epoch() # 重新开始一个epoch
        trainer.train(epoch, train_loader,  optimizer, training_phase=set_index + 1,
                      train_iters=len(train_loader), add_num=add_num, old_model=old_model,proto_type=proto_type
                      ) # 训练一个epoch
 
        lr_scheduler.step() # 学习率调整       
       

        if ((epoch + 1) % args.eval_epoch == 0 or epoch+1==Epochs): # 每eval_epoch评估一次
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': 0.,
            }, True, fpath=osp.join(args.logs_dir, '{}_checkpoint.pth.tar'.format(name)))

            logger_res.append('epoch: {}'.format(epoch + 1))
            
            mAP=0.
            if args.middle_test: # 中间评估
                mAP = test_model(model, all_train_sets, all_test_only_sets, set_index, logger_res=logger_res) # 评估模型                  
          
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': mAP,
            }, True, fpath=osp.join(args.logs_dir, '{}_checkpoint.pth.tar'.format(name)))    

    return model 

# 用于模型测试
def test_model(model, all_train_sets, all_test_sets, set_index,  logger_res=None):
    begin = 0
    evaluator = Evaluator(model) # 评估器 
        
    R1_all = [] # 存储每个数据集的R1
    mAP_all = [] # 存储每个数据集的mAP
    names='' # 存储数据集名称
    Results='' # 存储结果
    train_mAP=0 
    for i in range(begin, set_index + 1):
        dataset, num_classes, train_loader, test_loader, init_loader, name = all_train_sets[i] # 取出数据集 类别数 训练集加载器 测试集加载器 初始化集加载器 名称
        print('Results on {}'.format(name))

        train_R1, train_mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery,
                                                 cmc_flag=True)  # ,training_phase=i+1) # 评估R1和mAP
        R1_all.append(train_R1) # 添加R1
        mAP_all.append(train_mAP) # 添加mAP
        names = names + name + '\t\t' # 添加名称
        Results=Results+'|{:.1f}/{:.1f}\t'.format(train_mAP* 100, train_R1* 100) # 添加结果

    aver_mAP = torch.tensor(mAP_all).mean() # 计算平均mAP
    aver_R1 = torch.tensor(R1_all).mean() # 计算平均R1


    R1_all = [] # 存储每个数据集的R1
    mAP_all = [] # 存储每个数据集的mAP
    names_unseen = '' # 存储数据集名称
    Results_unseen = '' # 存储结果
    for i in range(len(all_test_sets)):
        dataset, num_classes, train_loader, test_loader, init_loader, name = all_test_sets[i] # 取出测试集 类别数 训练集加载器 测试集加载器 初始化集加载器 名称
        print('Results on {}'.format(name))
        R1, mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery,
                                     cmc_flag=True) # 评估R1和mAP

        R1_all.append(R1) # 添加R1
        mAP_all.append(mAP) # 添加mAP
        names_unseen = names_unseen + name + '\t' # 添加名称
        Results_unseen = Results_unseen + '|{:.1f}/{:.1f}\t'.format(mAP* 100, R1* 100) # 添加结果

    aver_mAP_unseen = torch.tensor(mAP_all).mean() # 计算平均mAP
    aver_R1_unseen = torch.tensor(R1_all).mean() # 计算平均R1
 
    print("Average mAP on Seen dataset: {:.1f}%".format(aver_mAP * 100))
    print("Average R1 on Seen dataset: {:.1f}%".format(aver_R1 * 100))
    names = names + '|Average\t|' # 添加名称
    Results = Results + '|{:.1f}/{:.1f}\t|'.format(aver_mAP * 100, aver_R1 * 100) # 添加结果
    print(names)
    print(Results)
    '''_________________________'''
    print("Average mAP on unSeen dataset: {:.1f}%".format(aver_mAP_unseen * 100))
    print("Average R1 on unSeen dataset: {:.1f}%".format(aver_R1_unseen * 100))
    names_unseen = names_unseen + '|Average\t|' # 添加名称
    Results_unseen = Results_unseen + '|{:.1f}/{:.1f}\t|'.format(aver_mAP_unseen* 100, aver_R1_unseen* 100) # 添加结果
    print(names_unseen)
    print(Results_unseen)
    if logger_res:
        logger_res.append(names)
        logger_res.append(Results)
        logger_res.append(Results.replace('|','').replace('/','\t'))
        logger_res.append(names_unseen)
        logger_res.append(Results_unseen)
        logger_res.append(Results_unseen.replace('|', '').replace('/', '\t'))
    return train_mAP # 返回最后一个训练集的mAP


# 用来融合新旧模型参数
def linear_combination(args, model, model_old, alpha, model_old_id=-1):
    '''old model '''
    model_old_state_dict = model_old.state_dict() # 获得旧模型参数
    '''latest trained model'''
    model_state_dict = model.state_dict() # 获得新模型参数

    ''''create new model'''
    model_new = copy.deepcopy(model) # 复制新模型
    model_new_state_dict = model_new.state_dict() # 获得新模型参数
    '''fuse the parameters'''
    for k, v in model_state_dict.items(): 
        if model_old_state_dict[k].shape == v.shape: # 如果新旧模型参数shape一样
            # print(k,'+++')
                model_new_state_dict[k] = alpha * v + (1 - alpha) * model_old_state_dict[k]
        else:
            print(k, '...')
            num_class_old = model_old_state_dict[k].shape[0] # 获得旧模型参数的第一个维度
            model_new_state_dict[k][:num_class_old] = alpha * v[:num_class_old] + (1 - alpha) * model_old_state_dict[k] # 融合旧模型参数
    model_new.load_state_dict(model_new_state_dict) # 加载融合后的参数
    return model_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model    
    parser.add_argument('--MODEL', type=str, default='50x',
                        choices=['50x'])
    # optimizer
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'],
                        help="optimizer ")
    parser.add_argument('--lr', type=float, default=0.008,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[30],
                        help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--resume', type=str, default=None, metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs0', type=int, default=80)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--eval_epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=200)
    
    # path   
    working_dir = osp.dirname(osp.abspath(__file__)) # 工作区路径
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                            default=osp.join(working_dir, '/data3/zhangshipeng/Database'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join('../logs/try'))

      
    parser.add_argument('--test_folder', type=str, default=None, help="test the models in a file")
   
    parser.add_argument('--setting', type=int, default=1, choices=[1, 2], help="training order setting")
    parser.add_argument('--middle_test', action='store_true', help="test during middle step")
    parser.add_argument('--AF_weight', default=0.1, type=float, help="anti-forgetting weight")   
    parser.add_argument('--n_sampling', default=6, type=int, help="number of sampling by Gaussian distribution")
    parser.add_argument('--lambda_1', default=0.1, type=float, help="temperature")
    parser.add_argument('--lambda_2', default=0.1, type=float, help="temperature")  
    main()
