'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world  # 全局配置模块
import numpy as np  # 数值计算库
import torch  # PyTorch 主库
import utils  # 工具函数模块
import dataloader  # 数据加载模块（主要用于类型标注）
from pprint import pprint  # 美观打印（本文件中未显式使用，保留原实现）
from utils import timer  # 计时器上下文管理器
from time import time  # 时间函数
from tqdm import tqdm  # 进度条库（本文件中未显式使用，保留原实现）
import model  # 模型定义模块（类型标注）
import multiprocessing  # 多进程库
from sklearn.metrics import roc_auc_score  # AUC 指标（本文件当前流程未启用）


# 测试多进程默认核心数：总核心的一半
CORES = multiprocessing.cpu_count() // 2


def _resolve_hnm_ratio(epoch):
    mode = world.config.get('hnm_mode', 'none')
    if mode == 'none':
        return 0.0
    if mode == 'mix':
        return float(world.config.get('hnm_ratio', 0.0))
    if mode == 'curriculum':
        start = float(world.config.get('hnm_curr_start', 0.0))
        end = float(world.config.get('hnm_curr_end', 0.0))
        warmup = int(world.config.get('hnm_warmup_epochs', 1))
        if warmup <= 1:
            return end
        progress = min(1.0, max(0.0, float(epoch) / float(warmup - 1)))
        return start + (end - start) * progress
    return 0.0


def _sample_hard_negatives(dataset, recmodel, batch_users):
    with torch.no_grad():
        rating = recmodel.getUsersRating(batch_users)
        batch_user_np = batch_users.detach().cpu().numpy()
        for idx, user in enumerate(batch_user_np):
            pos_items = dataset.allPos[user]
            if len(pos_items) > 0:
                rating[idx, pos_items] = -(1 << 10)
        hard_neg = torch.argmax(rating, dim=1)
    return hard_neg


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    # 当前训练模型对象
    Recmodel = recommend_model
    # 切换到训练模式（启用 dropout/bn 的训练行为）
    Recmodel.train()
    # BPR 损失封装对象
    bpr: utils.BPRLoss = loss_class

    # 负采样阶段计时
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)

    # 将采样结果转为 GPU/CPU 张量
    users = torch.as_tensor(S[:, 0], dtype=torch.long, device=world.device)
    posItems = torch.as_tensor(S[:, 1], dtype=torch.long, device=world.device)
    negItems = torch.as_tensor(S[:, 2], dtype=torch.long, device=world.device)

    # 打乱样本顺序
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)

    # 计算总 batch 数
    total_batch = len(users) // world.config['bpr_batch_size'] + 1

    # 累积平均损失
    aver_loss = 0.

    # 计算本轮 hard-negative 混采比例
    hnm_ratio = _resolve_hnm_ratio(epoch)
    hnm_ratio = min(1.0, max(0.0, hnm_ratio))

    # 逐 batch 训练
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        if hnm_ratio > 0.0:
            hard_neg = _sample_hard_negatives(dataset, Recmodel, batch_users)
            if hnm_ratio >= 1.0:
                batch_neg = hard_neg
            else:
                hard_mask = torch.rand(batch_neg.size(0), device=world.device) < hnm_ratio
                batch_neg = torch.where(hard_mask, hard_neg, batch_neg)

        # 单步优化（前向+反向+更新）
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        # 累加损失
        aver_loss += cri
        # 写入 TensorBoard
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)

    # 计算平均损失
    aver_loss = aver_loss / total_batch

    # 获取计时信息
    time_info = timer.dict()
    # 清空计时器
    timer.zero()
    # 返回日志字符串
    return f"loss{aver_loss:.3f}-hnm:{world.config.get('hnm_mode', 'none')}@{hnm_ratio:.3f}-{time_info}"
    
    
def test_one_batch(X):
    # 取出当前批次的 Top-K 排序结果（物品 id）
    sorted_items = X[0].numpy()
    # 取出当前批次真实测试集物品
    groundTrue = X[1]
    # 生成命中矩阵 r（命中为 1，否则 0）
    r = utils.getLabel(groundTrue, sorted_items)

    # 用于保存不同 K 下的指标
    pre, recall, ndcg = [], [], []

    # 遍历所有设定的 K
    for k in world.topks:
        # 计算 Recall / Precision
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        # 计算 NDCG
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))

    # 返回当前批次评估结果
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}
        
            
def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    # 测试用户批大小
    u_batch_size = world.config['test_u_batch_size']
    # 类型标注：数据集对象
    dataset: utils.BasicDataset
    # 测试字典：{user: [test_items]}
    testDict: dict = dataset.testDict
    # 类型标注：推荐模型
    Recmodel: model.LightGCN

    # 切换到评估模式（关闭 dropout）
    Recmodel = Recmodel.eval()

    # 取最大 K（用于一次 topk 后复用）
    max_K = max(world.topks)

    # 是否启用多进程评估
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)

    # 初始化最终结果累加器
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}

    # 评估阶段不需要梯度
    with torch.no_grad():
        # 所有需要评测的用户列表
        users = list(testDict.keys())

        # 简单检查测试 batch 是否过大
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")

        # 缓存每个 batch 的中间结果
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []

        # 理论 batch 数
        total_batch = len(users) // u_batch_size + 1

        # 分批评测用户
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            # 取当前批用户的训练集正样本（用于评测时过滤）
            allPos = dataset.getUserPosItems(batch_users)
            # 取当前批用户的测试真实物品
            groundTrue = [testDict[u] for u in batch_users]
            # 用户 id 转张量并放到设备上
            batch_users_gpu = torch.as_tensor(batch_users, dtype=torch.long, device=world.device)

            # 计算当前批用户对所有物品的预测分数
            rating = Recmodel.getUsersRating(batch_users_gpu)
            # rating = rating.cpu()

            # 构造需要过滤掉的训练交互索引
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)

            # 将训练集中已交互物品打成极小分，避免推荐到已见物品
            rating[exclude_index, exclude_items] = -(1 << 10)

            # 取每个用户的 top-K 预测物品
            _, rating_K = torch.topk(rating, k=max_K)

            # 转到 numpy（用于可选 AUC 计算等）
            rating = rating.cpu().numpy()
            # aucs = [
            #         utils.AUC(rating[i],
            #                   dataset,
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)

            # 释放临时内存
            del rating

            # 保存当前批次中间结果
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)

        # 校验 batch 数一致性
        assert total_batch == len(users_list)

        # 组合评估输入
        X = zip(rating_list, groundTrue_list)

        # 多进程或单进程计算 batch 指标
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))

        # 该变量在当前实现中未使用，保留原逻辑
        scale = float(u_batch_size / len(users))

        # 聚合所有 batch 的指标
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']

        # 对用户数做平均
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)

        # 写入 TensorBoard
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)

        # 关闭进程池
        if multicore == 1:
            pool.close()

        # 打印并返回评估结果
        print(results)
        return results
