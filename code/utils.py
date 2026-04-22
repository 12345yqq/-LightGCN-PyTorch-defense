# utils.py 中文注释版：尽量逐段解释，逻辑不变
'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import world  # 全局配置模块
import torch  # PyTorch 主库
from torch import nn, optim  # 神经网络组件与优化器
import numpy as np  # 数值计算库
from torch import log  # 对数函数（当前文件未显式使用，保留原结构）
from dataloader import BasicDataset  # 数据集基类（类型标注）
from time import time  # 时间函数
from model import LightGCN  # 模型类（当前文件中未显式使用，保留原结构）
from model import PairWiseModel  # 成对排序模型基类（类型标注）
from sklearn.metrics import roc_auc_score  # AUC 指标函数
import random  # 随机库（当前文件未显式使用，保留原结构）
import os  # 路径处理
try:
    # 尝试加载 C++ 负采样扩展（速度更快）
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    # C++ 源文件路径
    path = join(dirname(__file__), "sources/sampling.cpp")
    # 动态编译并导入扩展
    sampling = imp_from_filepath(path)
    # 同步扩展随机种子
    sampling.seed(world.seed)
    # 标记扩展可用
    sample_ext = True
except:
    # 扩展不可用则回退 Python 采样
    world.cprint("Cpp extension not loaded")
    sample_ext = False


class BPRLoss:
    def __init__(self,
                 recmodel : PairWiseModel,
                 config : dict):
        # 保存推荐模型
        self.model = recmodel
        # 保存 L2 权重衰减系数
        self.weight_decay = config['decay']
        # 保存学习率
        self.lr = config['lr']
        # 使用 Adam 优化器管理模型参数
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        # 计算模型返回的 BPR 主损失与正则损失
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        # 对正则项乘以权重衰减系数
        reg_loss = reg_loss*self.weight_decay
        # 得到总损失
        loss = loss + reg_loss

        # 清空历史梯度
        self.opt.zero_grad()
        # 反向传播
        loss.backward()
        # 参数更新
        self.opt.step()

        # 返回 Python 标量损失值
        return loss.cpu().item()


def UniformSample_original(dataset, neg_ratio = 1):
    # 类型标注
    dataset : BasicDataset
    # 获取每个用户正样本列表
    allPos = dataset.allPos
    # 记录采样开始时间
    start = time()
    # 优先使用 C++ 采样扩展
    if sample_ext:
        S = sampling.sample_negative(dataset.n_users, dataset.m_items,
                                     dataset.trainDataSize, allPos, neg_ratio)
    else:
        # 扩展不可用时使用 Python 实现
        S = UniformSample_original_python(dataset)
    # 返回采样结果 [user, pos, neg]
    return S

def UniformSample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    # 记录总耗时起点
    total_start = time()
    # 类型标注
    dataset : BasicDataset
    # 采样用户数量通常等于训练交互数
    user_num = dataset.trainDataSize
    # 在 [0, n_users) 范围随机采样用户
    users = np.random.randint(0, dataset.n_users, user_num)
    # 所有用户正样本索引
    allPos = dataset.allPos
    # 结果容器
    S = []
    # 两段耗时统计（保留原实现）
    sample_time1 = 0.
    sample_time2 = 0.
    # 遍历采样到的用户
    for i, user in enumerate(users):
        start = time()
        # 当前用户正样本集合
        posForUser = allPos[user]
        # 无正样本用户跳过
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        # 随机挑一个正样本
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        # 随机挑一个不在正样本中的负样本
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        # 记录三元组
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    # 总耗时（保留变量，便于调试）
    total = time() - total_start
    # 转为 numpy 数组返回
    return np.array(S)

# ===================end samplers==========================
# =====================utils====================================

def set_seed(seed):
    # 固定 numpy 随机种子
    np.random.seed(seed)
    # 仅当当前运行设备为 CUDA 时，才设置 CUDA 随机种子
    if world.device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 固定 CPU 端 torch 随机种子
    torch.manual_seed(seed)

def getFileName():
    # 根据模型名生成权重文件名
    if world.model_name == 'mf':
        file = f"mf-{world.dataset}-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == 'lgn':
        file = f"lgn-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == 'ngcf':
        file = f"ngcf-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}.pth.tar"
    else:
        raise NotImplementedError(f"Unsupported model_name in getFileName: {world.model_name}")
    # 拼接完整保存路径
    return os.path.join(world.FILE_PATH,file)

def minibatch(*tensors, **kwargs):

    # 读取批大小，默认取配置值
    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    # 单输入张量时，逐段切片
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        # 多输入张量时，同步切片后打包返回
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    # 是否需要额外返回打乱后的索引
    require_indices = kwargs.get('indices', False)

    # 检查所有输入长度是否一致
    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    # 生成并打乱索引
    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    # 根据输入数量返回打乱后的结果
    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    # 按需返回索引
    if require_indices:
        return result, shuffle_indices
    else:
        return result


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # 全局无名计时记录
    NAMED_TAPE = {}

    @staticmethod
    def get():
        # 读取并弹出最近一次无名计时
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        # 将命名计时汇总为可打印字符串
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        # 重置命名计时器
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        # 命名计时模式
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            # 无名计时模式
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        # 进入上下文时记录开始时间
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 退出上下文时累计耗时
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    # 命中数量（按用户）
    right_pred = r[:, :k].sum(1)
    # precision 分母 k
    precis_n = k
    # recall 分母为每个用户真实测试物品数
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    # 汇总 recall
    recall = np.sum(right_pred/recall_n)
    # 汇总 precision
    precis = np.sum(right_pred)/precis_n
    # 返回指标字典
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    # 取前 k 列命中结果
    pred_data = r[:, :k]
    # 倒数排名折扣权重
    scores = np.log2(1./np.arange(1, k+1))
    # 按位置加权
    pred_data = pred_data/scores
    # 对每个用户求和
    pred_data = pred_data.sum(1)
    # 汇总全部用户
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    # 保证 batch 对齐
    assert len(r) == len(test_data)
    # 取前 k 的预测命中矩阵
    pred_data = r[:, :k]

    # 构造理想命中矩阵（每个用户前 len(gt) 或 k 位为 1）
    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    # 计算理想 DCG
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    # 计算实际 DCG
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    # 避免除零
    idcg[idcg == 0.] = 1.
    # NDCG = DCG / IDCG
    ndcg = dcg/idcg
    # 将 NaN 置 0
    ndcg[np.isnan(ndcg)] = 0.
    # 汇总返回
    return np.sum(ndcg)

def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    # 类型标注
    dataset : BasicDataset
    # 构造全物品标签向量（测试真值为 1）
    r_all = np.zeros((dataset.m_items, ))
    r_all[test_data] = 1
    # 仅保留有效评分位置
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    # 返回二分类 AUC
    return roc_auc_score(r, test_item_scores)

def getLabel(test_data, pred_data):
    # 命中矩阵容器
    r = []
    for i in range(len(test_data)):
        # 当前用户真实测试物品
        groundTrue = test_data[i]
        # 当前用户预测 Top-K 物品
        predictTopK = pred_data[i]
        # 判断每个预测物品是否命中真实集合
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        # 转为 float 向量
        pred = np.array(pred).astype("float")
        r.append(pred)
    # 返回二维命中矩阵
    return np.array(r).astype('float')

# ====================end Metrics=============================
# =========================================================
