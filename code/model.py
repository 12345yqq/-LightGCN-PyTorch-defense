"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
# 导入全局配置模块（设备、超参数、日志打印函数等）
import world
# 导入 PyTorch 主库
import torch
# 导入数据集基础抽象类，用于类型标注
from dataloader import BasicDataset
# 导入神经网络模块别名 nn
from torch import nn
import torch.nn.functional as F
# 导入 numpy（当前文件中基本未使用，保留原实现一致性）
import numpy as np



# 基础模型抽象类：定义推荐模型统一接口
class BasicModel(nn.Module):    
    # 构造函数：继承 nn.Module 的初始化
    def __init__(self):
        super(BasicModel, self).__init__()
    
    # 给定一批用户，返回其对所有物品的评分（用于 Top-K 推荐）
    def getUsersRating(self, users):
        # 抽象方法：子类必须实现
        raise NotImplementedError
    
# PairWise（成对排序）模型抽象类：约定 bpr_loss 接口
class PairWiseModel(BasicModel):
    # 构造函数
    def __init__(self):
        super(PairWiseModel, self).__init__()

    # BPR 损失接口：输入用户、正样本、负样本
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        # 抽象方法：由具体模型实现
        raise NotImplementedError
    

# 纯矩阵分解模型（MF）：作为基线模型
class PureMF(BasicModel):
    # 初始化函数：接收配置和数据集
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        # 先初始化父类
        super(PureMF, self).__init__()
        # 用户数量
        self.num_users  = dataset.n_users
        # 物品数量
        self.num_items  = dataset.m_items
        # 隐向量维度
        self.latent_dim = config['latent_dim_rec']
        # 评分输出激活函数（将分数压到 0~1）
        self.f = nn.Sigmoid()
        # 初始化参数
        self.__init_weight()
        
    # 初始化用户和物品的嵌入参数
    def __init_weight(self):
        # 用户嵌入矩阵：形状 [num_users, latent_dim]
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        # 物品嵌入矩阵：形状 [num_items, latent_dim]
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        # 打印初始化信息
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    # 全排序评分：输入用户集合，输出对全体物品的评分
    def getUsersRating(self, users):
        # 保证索引类型为 long
        users = users.long()
        # 取出用户向量
        users_emb = self.embedding_user(users)
        # 使用全部物品嵌入
        items_emb = self.embedding_item.weight
        # 矩阵乘法得到 [batch_user, num_items] 分数矩阵
        scores = torch.matmul(users_emb, items_emb.t())
        # 经过 Sigmoid 输出
        return self.f(scores)
    
    # BPR 损失：让正样本分数高于负样本分数
    def bpr_loss(self, users, pos, neg):
        # 取用户嵌入
        users_emb = self.embedding_user(users.long())
        # 取正样本物品嵌入
        pos_emb   = self.embedding_item(pos.long())
        # 取负样本物品嵌入
        neg_emb   = self.embedding_item(neg.long())
        # 正样本打分（点积）
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        # 负样本打分（点积）
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        # 排序损失：softplus(neg - pos)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        # L2 正则：约束嵌入幅值，减轻过拟合
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        # 返回主损失与正则损失（外部会组合）
        return loss, reg_loss
        
    # 前向函数：输入用户-物品配对，输出单点评分
    def forward(self, users, items):
        # 转换索引类型
        users = users.long()
        # 转换索引类型
        items = items.long()
        # 用户嵌入
        users_emb = self.embedding_user(users)
        # 物品嵌入
        items_emb = self.embedding_item(items)
        # 点积得到交互分数
        scores = torch.sum(users_emb*items_emb, dim=1)
        # Sigmoid 输出
        return self.f(scores)


# LightGCN 主模型：图协同过滤核心实现regis
class LightGCN(BasicModel):
    # 初始化函数：保存配置并初始化参数
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        # 父类初始化
        super(LightGCN, self).__init__()
        # 保存配置字典
        self.config = config
        # 保存数据集对象（注：这里沿用原代码类型标注写法）
        self.dataset : dataloader.BasicDataset = dataset
        # 初始化权重和图
        self.__init_weight()

    # 初始化 LightGCN 所需参数与图结构
    def __init_weight(self):
        # 用户数量
        self.num_users  = self.dataset.n_users
        # 物品数量
        self.num_items  = self.dataset.m_items
        # 隐向量维度
        self.latent_dim = self.config['latent_dim_rec']
        # 图传播层数
        self.n_layers = self.config['lightGCN_n_layers']
        # 层聚合方式：mean(原版) 或 learnable(可学习加权)
        self.layer_agg = self.config.get('layer_agg', 'mean')
        # 图 dropout 保留概率
        self.keep_prob = self.config['keep_prob']
        # 是否采用分块邻接矩阵（超大图场景）
        self.A_split = self.config['A_split']
        # 用户嵌入表
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        # 物品嵌入表
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        # 未使用预训练参数时，进行随机初始化
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            # 正态分布初始化（原作者实现）
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            # 正态分布初始化（原作者实现）
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            # 打印提示
            world.cprint('use NORMAL distribution initilizer')
        # 使用预训练参数分支
        else:
            # 拷贝预训练用户嵌入
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            # 拷贝预训练物品嵌入
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            # 打印提示
            print('use pretarined data')
        # 评分激活函数
        self.f = nn.Sigmoid()
        # 获取归一化稀疏图（由 dataloader 负责构建/加载）
        self.Graph = self.dataset.getSparseGraph()
        
        # 当启用 learnable 聚合时，才创建可学习层权重
        if self.layer_agg == 'learnable':
            # 为每一层（含第0层，共 n_layers + 1 层）分配可学习权重
            self.layer_weights = nn.Parameter(torch.ones(self.n_layers + 1))
        
        # 打印模型准备完成信息
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")

    # 对单个稀疏矩阵执行边 dropout
    def __dropout_x(self, x, keep_prob):
        # 保存原矩阵尺寸
        size = x.size()
        # 取稀疏索引（转置后每行是一条边）
        index = x.indices().t()
        # 取每条边对应权重
        values = x.values()
        # 生成随机掩码：保留概率约为 keep_prob
        random_index = torch.rand(len(values), device=values.device) + keep_prob
        # 转为布尔掩码
        random_index = random_index.int().bool()
        # 过滤保留的边索引
        index = index[random_index]
        # 保留边权重并做反缩放，保持期望不变
        values = values[random_index]/keep_prob
        # 重建稀疏矩阵（保持与原图同设备，避免隐式回退到 CPU）
        g = torch.sparse_coo_tensor(index.t(), values, size, device=values.device).coalesce()
        # 返回 dropout 后图
        return g
    
    # 对图执行 dropout：支持分块和整图两种模式
    def __dropout(self, keep_prob):
        # 分块图模式
        if self.A_split:
            # 存放每一块 dropout 结果
            graph = []
            # 逐块处理
            for g in self.Graph:
                # 对当前块执行 dropout
                graph.append(self.__dropout_x(g, keep_prob))
        # 整图模式
        else:
            # 对整张图执行 dropout
            graph = self.__dropout_x(self.Graph, keep_prob)
        # 返回处理后的图
        return graph
    
    # LightGCN 传播主函数：返回最终用户/物品表示
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        # 取用户初始嵌入（第 0 层）
        users_emb = self.embedding_user.weight
        # 取物品初始嵌入（第 0 层）
        items_emb = self.embedding_item.weight
        # 拼接成统一节点表示 [num_users + num_items, dim]
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        # 保存每一层表示，embs[0] 是 0 层（ego embedding）
        embs = [all_emb]
        # 若开启图 dropout
        if self.config['dropout']:
            # 仅在训练阶段做 dropout
            if self.training:
                # 打印提示
                print("droping")
                # 获取 dropout 后图
                g_droped = self.__dropout(self.keep_prob)
            # 测试阶段不做 dropout
            else:
                # 直接使用原图
                g_droped = self.Graph        
        # 不开启 dropout
        else:
            # 直接使用原图
            g_droped = self.Graph    
        
        # 逐层传播，共 n_layers 层
        for layer in range(self.n_layers):
            # 分块图传播分支
            if self.A_split:
                # 临时保存每块传播结果
                temp_emb = []
                # 遍历每一块子图
                for f in range(len(g_droped)):
                    # 稀疏矩阵乘法：邻接矩阵 * 上一层表示
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                # 将所有块在节点维拼接
                side_emb = torch.cat(temp_emb, dim=0)
                # 更新当前层表示
                all_emb = side_emb
            # 非分块图传播分支
            else:
                # 稀疏矩阵乘法完成一次图卷积传播
                all_emb = torch.sparse.mm(g_droped, all_emb)
            # 保存该层表示
            embs.append(all_emb)
        # 堆叠为三维张量：[num_nodes, n_layers+1, dim]
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        
        # 层聚合：支持原版 mean 与改进版 learnable 两种模式
        if self.layer_agg == 'learnable':
            tau = max(float(self.config.get('layer_w_tau', 1.0)), 1e-6)
            # 先对可学习权重做 softmax，保证权重和为 1
            alpha = torch.softmax(self.layer_weights / tau, dim=0)
            # 扩展维度到 [1, n_layers+1, 1] 以进行广播乘法
            alpha = alpha.view(1, -1, 1)
            # 按层加权求和
            light_out = torch.sum(embs * alpha, dim=1)
        else:
            # 原始 LightGCN：各层简单平均
            light_out = torch.mean(embs, dim=1)
        
        # 按节点类型拆回用户表示和物品表示
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        # 返回最终用户、物品向量
        return users, items
    
    # 全排序评分：给定用户集合，对全体物品打分
    def getUsersRating(self, users):
        # 获取传播后的用户和物品向量
        all_users, all_items = self.computer()
        # 取当前 batch 用户向量
        users_emb = all_users[users.long()]
        # 全部物品向量
        items_emb = all_items
        # 矩阵乘法得到用户对所有物品分数
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        # 返回评分矩阵
        return rating
    
    # 返回 BPR 训练所需表示（传播后 + 初始嵌入）
    def getEmbedding(self, users, pos_items, neg_items):
        # 计算传播后表示
        all_users, all_items = self.computer()
        # 用户传播后向量
        users_emb = all_users[users]
        # 正样本传播后向量
        pos_emb = all_items[pos_items]
        # 负样本传播后向量
        neg_emb = all_items[neg_items]
        # 用户第 0 层向量（用于正则）
        users_emb_ego = self.embedding_user(users)
        # 正样本第 0 层向量（用于正则）
        pos_emb_ego = self.embedding_item(pos_items)
        # 负样本第 0 层向量（用于正则）
        neg_emb_ego = self.embedding_item(neg_items)
        # 一并返回
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    # LightGCN 的 BPR 损失计算
    def bpr_loss(self, users, pos, neg):
        # 取传播后向量与第 0 层向量
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        # L2 正则项（只对第 0 层参数做约束）
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        # 正样本逐维乘积
        pos_scores = torch.mul(users_emb, pos_emb)
        # 正样本点积分数
        pos_scores = torch.sum(pos_scores, dim=1)
        # 负样本逐维乘积
        neg_scores = torch.mul(users_emb, neg_emb)
        # 负样本点积分数
        neg_scores = torch.sum(neg_scores, dim=1)
        
        # BPR 主损失：希望 pos_scores > neg_scores
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        # 轻量正则化对照：仅在 learnable 聚合时对层权重添加小 L2 惩罚
        layer_w_l2 = self.config.get('layer_w_l2', 0.0)
        if self.layer_agg == 'learnable' and layer_w_l2 > 0:
            reg_loss = reg_loss + layer_w_l2 * torch.sum(self.layer_weights.pow(2))
        
        # 返回主损失和正则损失
        return loss, reg_loss
       
    # 前向接口：给定用户-物品配对，输出匹配分数
    def forward(self, users, items):
        # compute embedding
        # 先得到传播后的最终用户/物品向量
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        # 取当前用户向量
        users_emb = all_users[users]
        # 取当前物品向量
        items_emb = all_items[items]
        # 逐维相乘
        inner_pro = torch.mul(users_emb, items_emb)
        # 沿隐向量维求和，得到最终打分
        gamma     = torch.sum(inner_pro, dim=1)
        # 返回打分
        return gamma


class NGCF(BasicModel):
    def __init__(self, config: dict, dataset: BasicDataset):
        super(NGCF, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']

        self.embedding_user = nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_item = nn.Embedding(self.num_items, self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.W_gc = nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.n_layers)])
        self.W_bi = nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.n_layers)])
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.f = nn.Sigmoid()

        self.Graph = self.dataset.getSparseGraph()
        print(f"ngcf is already to go(dropout:{self.config['dropout']})")

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        ego_embeddings = torch.cat([users_emb, items_emb], dim=0)

        all_embeddings = [ego_embeddings]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], ego_embeddings))
                side_embeddings = torch.cat(temp_emb, dim=0)
            else:
                side_embeddings = torch.sparse.mm(g_droped, ego_embeddings)

            sum_embeddings = self.act(self.W_gc[layer](side_embeddings))
            bi_embeddings = self.act(self.W_bi[layer](ego_embeddings * side_embeddings))
            ego_embeddings = sum_embeddings + bi_embeddings

            if self.training and self.config['dropout']:
                msg_drop_prob = 1.0 - float(self.keep_prob)
                msg_drop_prob = min(max(msg_drop_prob, 0.0), 0.9)
                ego_embeddings = F.dropout(ego_embeddings, p=msg_drop_prob, training=True)

            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings.append(norm_embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=1)
        users, items = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        rating = self.f(torch.matmul(users_emb, all_items.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        users = users.long()
        pos = pos.long()
        neg = neg.long()
        users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0 = self.getEmbedding(users, pos, neg)

        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        return loss, reg_loss

    def forward(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items[items.long()]
        gamma = torch.sum(users_emb * items_emb, dim=1)
        return gamma
