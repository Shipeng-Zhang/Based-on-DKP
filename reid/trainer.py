from __future__ import print_function, absolute_import
import time

from torch.nn import functional as F
import torch
import torch.nn as nn

from reid.loss.loss_uncertrainty import TripletLoss_set
from .utils.meters import AverageMeter
from reid.metric_learning.distance import cosine_similarity

class Trainer(object):
    def __init__(self,args, model, writer=None):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.writer = writer
        self.uncertainty=True # 使用不确定性估计
        if self.uncertainty:
            self.criterion_ce=nn.CrossEntropyLoss() # 交叉熵损失函数，用来保证采样特征与中心特征的一致性  IDM
            self.criterion_triple=TripletLoss_set() # 三元组损失函数，用来保证同一特征的聚集和特征之间区分 IDM

        
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean') # KL散度损失函数，用来保证新数据的差异性且旧数据不被遗忘 PKT
        
        self.AF_weight=args.AF_weight   # anti-forgetting loss 
             
        self.n_sampling=args.n_sampling    # 每个样本采样的次数     
       
    def train(self, epoch, data_loader_train,  optimizer, training_phase,
              train_iters=200, add_num=0, old_model=None, proto_type=None        
              ):       
        self.model.train() # 训练模式
        # freeze the bn layer totally
        for m in self.model.module.base.modules():
            if isinstance(m, nn.BatchNorm2d): # 如果是BN层
                if m.weight.requires_grad == False and m.bias.requires_grad == False: # 如果BN层的权重和偏置不需要梯度
                    m.eval() # 设置为评估模式
        if proto_type is not None: # 使用原型
            proto_type_merge={}  # 合并所有原型
            steps=list(proto_type.keys())
            steps.sort()
            stages=1
            if stages<len(steps):
                steps=steps[-stages:]

            proto_type_merge['mean_features']=torch.cat([proto_type[k]['mean_features'] for k in steps]) # 合并所有原型特征的均值
            proto_type_merge['labels'] = torch.tensor([proto_type[k]['labels'] for k in steps]).to(proto_type_merge['mean_features'].device) # 合并所有原型的标签
            proto_type_merge['mean_vars'] = torch.cat([proto_type[k]['mean_vars'] for k in steps]) # 合并所有原型特征的方差
           
            features_mean=proto_type_merge['mean_features'] # 原型特征的均值

        batch_time = AverageMeter() # 计算每个batch的时间
        data_time = AverageMeter() # 计算每个batch的数据加载时间
        losses_ce = AverageMeter() # 交叉熵损失
        losses_tr = AverageMeter() # 三元组损失

        end = time.time() # 记录时间

        for i in range(train_iters):
            train_inputs = data_loader_train.next() # 读取数据 imgs
            data_time.update(time.time() - end) # 计算数据加载时间

            # 处理数据 返回数据图片 标签 相机 域
            s_inputs, targets, cids = self._parse_data(train_inputs)
            targets += add_num # 标签偏移
            # BS*2048, BS*(1+n_sampling)*2048,BS*(1+n_sampling)*num_classes, BS*2048
            s_features, merge_feat, cls_outputs, out_var,feat_final_layer = self.model(s_inputs) # 前向传播 获得特征 合并特征 分类输出 方差

            loss_ce, loss_tp = 0, 0 # 交叉熵损失 三元组损失
            if self.uncertainty: # 使用不确定性估计
                ###ID loss###
                for s_id in range(1 + self.n_sampling):
                    loss_ce += self.criterion_ce(cls_outputs[:, s_id], targets) # 计算交叉熵损失
                loss_ce = loss_ce / (1 + self.n_sampling) # 取平均
                loss_ce=loss_ce*1 # 交叉熵损失权重1
                ###set triplet-loss##
                loss_tp = self.criterion_triple(merge_feat, targets)[0] # 计算三元组损失
                loss_tp=loss_tp*1.5 # 三元组损失权重1.5           

            
            loss = loss_ce + loss_tp  # 损失L = 交叉熵损失 + 三元组损失       

            losses_ce.update(loss_ce.item()) # 记录交叉熵损失
            losses_tr.update(loss_tp.item()) # 记录三元组损失


            if training_phase>1: # 第二阶段开始使用PKT
                divergence=0.                       
                center_feat=merge_feat[:,0]  # obtain the center feature 
                Affinity_matrix_new = self.get_normal_affinity(center_feat, self.args.lambda_2) # obtain the affinity matrix
                features_var = proto_type_merge['mean_vars']    # obtain the prototype strandard variances
                noises = torch.randn(features_mean.size()).to(features_mean.device) # generate gaussian noise
                samples = noises * features_var + features_mean # obtain noised sample
                samples = F.normalize(samples, dim=1)   # normalize the sample
                s_features_old = cosine_similarity(center_feat, samples)    # obtain the new-old relation
                s_features_old = F.softmax(s_features_old /self.args.lambda_1, dim=1)  # normalize the relation

                Affinity_matrix_old = self.get_normal_affinity(s_features_old, self.args.lambda_2)  # obtain the affinity matrix under the prototype view
                # divergence += self.cal_KL(Affinity_matrix_new, Affinity_matrix_old, targets)
                Affinity_matrix_new_log = torch.log(Affinity_matrix_new)
                divergence+=self.KLDivLoss(Affinity_matrix_new_log, Affinity_matrix_old)
                
                loss = loss + divergence * self.AF_weight                               
            
            optimizer.zero_grad()

            loss.backward()
            
                     
            optimizer.step() # 更新参数
           
            batch_time.update(time.time() - end) # 计算每个batch的时间
            end = time.time() # 记录时间
            if self.writer != None :
                self.writer.add_scalar(tag="loss/Loss_ce_{}".format(training_phase), scalar_value=losses_ce.val,
                          global_step=epoch * train_iters + i)
                self.writer.add_scalar(tag="loss/Loss_tr_{}".format(training_phase), scalar_value=losses_tr.val,
                          global_step=epoch * train_iters + i)

                self.writer.add_scalar(tag="time/Time_{}".format(training_phase), scalar_value=batch_time.val,
                          global_step=epoch * train_iters + i)
            if (i + 1) == train_iters: # 最后一次迭代
            #if 1 :
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tp {:.3f} ({:.3f})\t'                     
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              ))
        


    def get_normal_affinity(self,x,Norm=0.1):
        pre_matrix_origin=cosine_similarity(x,x)
        pre_affinity_matrix=F.softmax(pre_matrix_origin/Norm, dim=1)
        return pre_affinity_matrix




    def _parse_data(self, inputs):
        imgs, _,pids, cids = inputs
        if not torch.is_tensor(pids):
            pids = torch.tensor(pids, dtype=torch.long)
        if not torch.is_tensor(cids):
            cids = torch.tensor(cids, dtype=torch.long)
        return imgs.cuda(), pids.cuda(), cids.cuda()



