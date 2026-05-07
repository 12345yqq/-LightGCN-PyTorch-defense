'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''

import os  # 操作系统与路径相关
from os.path import join  # 路径拼接工具
import torch  # PyTorch 库
from enum import Enum  # 枚举类型（当前文件中未实际使用，保留原结构）
from parse import parse_args  # 命令行参数解析函数
import multiprocessing  # 多进程工具（用于获取 CPU 核心数）

# 允许部分 OpenMP 重复加载场景，避免某些环境报错
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 读取命令行参数
args = parse_args()

# 项目根目录
ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
# 代码目录
CODE_PATH = join(ROOT_PATH, 'code')
# 数据目录
DATA_PATH = join(ROOT_PATH, 'data')
# TensorBoard 日志目录
BOARD_PATH = join(CODE_PATH, 'runs')
# 权重保存目录
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys  # 系统路径操作
# 将 C++ 扩展源码目录加入模块搜索路径
sys.path.append(join(CODE_PATH, 'sources'))


# 若权重目录不存在则创建
if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


# 用于保存训练和模型超参数的全局配置字典
config = {}

# 支持的数据集与模型列表
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book']
all_models = ['mf', 'lgn', 'ngcf']

# 将命令行参数写入配置字典
# config['batch_size'] = 4096
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers'] = args.layer
config['layer_agg'] = args.layer_agg
config['layer_w_l2'] = args.layer_w_l2
config['layer_w_tau'] = args.layer_w_tau
config['hnm_mode'] = args.hnm_mode
config['hnm_ratio'] = args.hnm_ratio
config['hnm_curr_start'] = args.hnm_curr_start
config['hnm_curr_end'] = args.hnm_curr_end
config['hnm_warmup_epochs'] = args.hnm_warmup_epochs
config['dropout'] = args.dropout
config['keep_prob'] = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['cl_weight'] = args.cl_weight
config['cl_temp'] = args.cl_temp
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False

# 判断是否实际启用 GPU：既要用户允许，也要本机可用
GPU = (args.cuda == 1) and torch.cuda.is_available()

# 若启用 GPU，设置目标设备并开启 cuDNN benchmark（固定输入形状时可加速）
if GPU:
    torch.cuda.set_device(args.gpu_id)
    torch.backends.cudnn.benchmark = True

# 统一设备对象
device = torch.device(f'cuda:{args.gpu_id}' if GPU else "cpu")
print(f"[Device] {device}")

# 测试多进程默认可用核心数
CORES = multiprocessing.cpu_count() // 2

# 全局随机种子
seed = args.seed

# 数据集名称
dataset = args.dataset
# 模型名称
model_name = args.model

# 参数合法性检查：数据集
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")

# 参数合法性检查：模型
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")




# 训练总轮数
TRAIN_epochs = args.epochs
# 测试间隔（至少为 1）
TEST_INTERVAL = max(1, args.test_interval)
# 保存间隔（至少为 1）
SAVE_INTERVAL = max(1, args.save_interval)
# 是否加载权重
LOAD = args.load
# 权重路径参数
PATH = args.path
# Top-K 指标列表（字符串转为 Python 列表）
topks = eval(args.topks)
# 是否启用 TensorBoard
tensorboard = args.tensorboard
# 实验注释
comment = args.comment
# 屏蔽 FutureWarning，减少控制台噪声
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words: str):
    # 以黄色高亮背景打印日志，便于识别关键信息
    print(f"\033[0;30;43m{words}\033[0m")

logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""
# 字体来源：ANSI Shadow
# 参考：http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
# print(logo)  # 如需显示可取消注释
