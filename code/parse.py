'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse  # 命令行参数解析库


def parse_args():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Go lightGCN")

    # 是否在可用时启用 CUDA（1=启用，0=禁用）
    parser.add_argument('--cuda', type=int, default=1,
                        help='whether to use cuda when available')
    # 指定使用哪张 GPU
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='gpu device id')
    # BPR 训练批大小
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    # 推荐嵌入维度
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    # LightGCN 传播层数
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
    # LightGCN 层聚合方式：mean(原版平均) 或 learnable(可学习加权)
    parser.add_argument('--layer_agg', type=str, default='mean', choices=['mean', 'learnable'],
                        help="layer aggregation type for LightGCN: mean or learnable")
    # learnable 层权重的额外 L2 正则系数（仅在 layer_agg=learnable 时生效）
    parser.add_argument('--layer_w_l2', type=float, default=0.0,
                        help='extra L2 regularization for learnable layer weights')
    # learnable 层权重 softmax 的温度系数（>0，默认 1.0）
    parser.add_argument('--layer_w_tau', type=float, default=1.0,
                        help='temperature for learnable layer-weight softmax')
    # Hard Negative Mix 模式：none(关闭)、mix(固定比例)、curriculum(随 epoch 线性增长)
    parser.add_argument('--hnm_mode', type=str, default='none', choices=['none', 'mix', 'curriculum'],
                        help='hard negative mixing mode')
    # 固定 HNM 比例（仅 hnm_mode=mix 时使用）
    parser.add_argument('--hnm_ratio', type=float, default=0.4,
                        help='hard negative ratio when hnm_mode=mix')
    # curriculum 起始比例
    parser.add_argument('--hnm_curr_start', type=float, default=0.2,
                        help='curriculum hard-negative ratio at epoch 0')
    # curriculum 结束比例
    parser.add_argument('--hnm_curr_end', type=float, default=0.6,
                        help='curriculum hard-negative ratio at warmup end')
    # curriculum 达到终点比例所需 epoch 数
    parser.add_argument('--hnm_warmup_epochs', type=int, default=150,
                        help='epochs to linearly increase hard-negative ratio in curriculum mode')
    # 学习率
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    # L2 正则权重（weight decay）
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    # SimGCL: 图对比学习损失权重
    parser.add_argument('--cl_weight', type=float, default=0.0,
                        help="weight of contrastive learning loss (SimGCL). Default 0.0 means disabled. Try 0.1 or 0.2")
    # SimGCL: InfoNCE 温度系数
    parser.add_argument('--cl_temp', type=float, default=0.2,
                        help="temperature for contrastive learning softmax")
    # 是否启用图 dropout
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    # dropout 保留概率
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    # 邻接矩阵分块数（大图时可用）
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    # 测试阶段用户批大小
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    # 数据集名称
    parser.add_argument('--dataset', type=str,default='gowalla',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    # 权重保存路径
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    # Top-K 评估列表（字符串形式）
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    # 是否启用 TensorBoard
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    # 实验备注名
    parser.add_argument('--comment', type=str,default="lgn")
    # 是否加载已有权重
    parser.add_argument('--load', type=int,default=0)
    # 总训练轮数
    parser.add_argument('--epochs', type=int,default=1000)
    # 每 N 轮做一次评估
    parser.add_argument('--test_interval', type=int, default=10,
                        help='run evaluation every N epochs')
    # 每 N 轮保存一次模型
    parser.add_argument('--save_interval', type=int, default=1,
                        help='save checkpoint every N epochs')
    # 测试是否启用多进程
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    # 是否使用预训练参数
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    # 随机种子
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    # 模型名称：mf / lgn / ngcf
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn, ngcf]')
    # 可选：指定权重文件名（相对于 code/checkpoints/）
    parser.add_argument('--ckpt', type=str, default=None,
                        help='checkpoint file name under code/checkpoints (optional)')

    # 返回解析后的参数对象
    return parser.parse_args()
