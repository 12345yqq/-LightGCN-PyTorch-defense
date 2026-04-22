# dataloader.py 中文注释版：尽量逐段解释，逻辑不变
"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import os  # 操作系统与路径处理
from os.path import join  # 路径拼接函数
import sys  # Python 系统模块（当前文件未显式使用，保留原结构）
import torch  # PyTorch 主库
import numpy as np  # 数值计算库
import pandas as pd  # 表格数据读取
from torch.utils.data import Dataset, DataLoader  # 数据集抽象类与加载器（DataLoader当前未显式使用）
from scipy.sparse import csr_matrix  # CSR 稀疏矩阵
import scipy.sparse as sp  # SciPy 稀疏矩阵库
import world  # 全局配置模块
from world import cprint  # 彩色打印
from time import time  # 时间函数

class BasicDataset(Dataset):
    def __init__(self):
        # 基类初始化提示
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class LastFM(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    LastFM dataset
    """
    def __init__(self, path="../data/lastfm"):
        # 输出正在加载的数据集名称
        cprint("loading [last fm]")
        # 数据模式映射：训练/测试
        self.mode_dict = {'train':0, "test":1}
        # 默认训练模式
        self.mode    = self.mode_dict['train']
        # self.n_users = 1892
        # self.m_items = 4489
        # 读取训练集交互数据（用户-物品）
        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        # print(trainData.head())
        # 读取测试集交互数据
        testData  = pd.read_table(join(path, 'test1.txt'), header=None)
        # print(testData.head())
        # 读取社交网络边（用户-用户）
        trustNet  = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()
        # print(trustNet[:5])
        # 原始 LastFM 下标从 1 开始，这里统一转为从 0 开始
        trustNet -= 1
        trainData-= 1
        testData -= 1
        # 保存原始数据引用
        self.trustNet  = trustNet
        self.trainData = trainData
        self.testData  = testData
        # 提取训练用户列
        self.trainUser = np.array(trainData[:][0])
        # 去重后的训练用户集合
        self.trainUniqueUsers = np.unique(self.trainUser)
        # 提取训练物品列
        self.trainItem = np.array(trainData[:][1])
        # self.trainDataSize = len(self.trainUser)
        # 提取测试用户列
        self.testUser  = np.array(testData[:][0])
        # 去重后的测试用户集合
        self.testUniqueUsers = np.unique(self.testUser)
        # 提取测试物品列
        self.testItem  = np.array(testData[:][1])
        # 图缓存（延迟构建）
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")
        
        # 用户-用户社交图
        self.socialNet    = csr_matrix((np.ones(len(trustNet)), (trustNet[:,0], trustNet[:,1]) ), shape=(self.n_users,self.n_users))
        # 用户-物品二部图（训练交互）
        self.UserItemNet  = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem) ), shape=(self.n_users,self.m_items)) 
        
        # 预计算每个用户的正样本物品
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        # 保存每个用户负样本全集（仅 LastFM 这种小数据集可行）
        self.allNeg = []
        allItems    = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        # 构建测试字典
        self.__testDict = self.__build_test()

    @property
    def n_users(self):
        return 1892
    
    @property
    def m_items(self):
        return 4489
    
    @property
    def trainDataSize(self):
        return len(self.trainUser)
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def getSparseGraph(self):
        # 若尚未构建图，则进行构建
        if self.Graph is None:
            # 用户索引张量
            user_dim = torch.LongTensor(self.trainUser)
            # 物品索引张量
            item_dim = torch.LongTensor(self.trainItem)
            
            # 上半块边：user -> item(+n_users)
            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            # 下半块边：item(+n_users) -> user
            second_sub = torch.stack([item_dim+self.n_users, user_dim])
            # 合并双向边
            index = torch.cat([first_sub, second_sub], dim=1)
            # 边权先置 1
            data = torch.ones(index.size(-1)).int()
            # 先构建整型稀疏图
            self.Graph = torch.sparse_coo_tensor(
                index,
                data,
                torch.Size([self.n_users + self.m_items, self.n_users + self.m_items]),
                dtype=torch.int32
            )
            # 转为稠密矩阵做归一化（LastFM 规模较小可以接受）
            dense = self.Graph.to_dense()
            # 计算节点度
            D = torch.sum(dense, dim=1).float()
            # 避免除零
            D[D==0.] = 1.
            # 度的平方根
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            # 左右归一化：D^{-1/2} A D^{-1/2}
            dense = dense/D_sqrt
            dense = dense/D_sqrt.t()
            # 取非零索引
            index = dense.nonzero()
            # 取非零值（过滤过小值）
            data  = dense[dense >= 1e-9]
            assert len(index) == len(data)
            # 转回 float 稀疏张量
            self.Graph = torch.sparse_coo_tensor(
                index.t(),
                data,
                torch.Size([self.n_users + self.m_items, self.n_users + self.m_items]),
                dtype=torch.float32
            )
            # 合并重复索引并迁移到设备
            self.Graph = self.Graph.coalesce().to(world.device)
        # 返回缓存图
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        # 使用字典聚合每个用户对应的测试物品
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        # 返回测试映射
        return test_data
    
    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))
    
    def getUserPosItems(self, users):
        # 收集每个用户的正样本物品列表
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems
            
    
    
    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user
    
    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']
    
    def __len__(self):
        return len(self.trainUniqueUsers)

class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self,config = world.config,path="../data/gowalla"):
        # 打印当前加载路径
        cprint(f'loading [{path}]')
        # 是否对邻接矩阵分块
        self.split = config['A_split']
        # 分块数量
        self.folds = config['A_n_fold']
        # 训练/测试模式映射
        self.mode_dict = {'train': 0, "test": 1}
        # 默认训练模式
        self.mode = self.mode_dict['train']
        # 用户总数（读取后更新）
        self.n_user = 0
        # 物品总数（读取后更新）
        self.m_item = 0
        # 训练文件路径
        train_file = path + '/train.txt'
        # 测试文件路径
        test_file = path + '/test.txt'
        # 保存数据集目录
        self.path = path
        # 用于缓存训练和测试解析结果
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        # 训练交互数
        self.traindataSize = 0
        # 测试交互数
        self.testDataSize = 0

        # 读取 train.txt：每行格式 user item1 item2 ...
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    # 当前用户的所有正样本物品
                    items = [int(i) for i in l[1:]]
                    # 当前用户 id
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    # 动态更新最大物品 id
                    self.m_item = max(self.m_item, max(items))
                    # 动态更新最大用户 id
                    self.n_user = max(self.n_user, uid)
                    # 累积训练交互数
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        # 读取 test.txt：格式同 train
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        # 因为 id 从 0 开始，最大 id + 1 才是数量
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        
        # 邻接图缓存
        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # 构建用户-物品交互稀疏矩阵
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        # 用户度（每个用户交互条数）
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        # 物品度（每个物品被交互条数）
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # 预计算每个用户正样本列表
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        # 构建测试字典
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self,A):
        # 将大邻接矩阵按行切分成多个子块
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce())
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        # scipy 稀疏矩阵转 torch 稀疏张量
        coo = X.tocoo().astype(np.float32)
        row = torch.from_numpy(coo.row.astype(np.int64)).to(world.device)
        col = torch.from_numpy(coo.col.astype(np.int64)).to(world.device)
        index = torch.stack([row, col], dim=0)
        data = torch.from_numpy(coo.data).to(dtype=torch.float32, device=world.device)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape), device=world.device)
        
    def getSparseGraph(self):
        # 获取（或构建）归一化邻接矩阵
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                # 优先加载预计算好的归一化邻接矩阵
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                # 若不存在缓存则现场构建
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                # 上右块写入 R
                adj_mat[:self.n_users, self.n_users:] = R
                # 下左块写入 R^T
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                
                # 计算度并构造 D^{-1/2}
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                # 对称归一化
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                # 将归一化矩阵缓存到磁盘
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                # 分块模式
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                # 整图模式
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce()
                print("don't split the matrix")
        # 返回图
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        # 按用户返回其正样本物品集合
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems
