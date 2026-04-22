import os  # 操作系统路径处理
import world  # 全局配置模块
import dataloader  # 数据加载模块
import model  # 模型定义模块
from pprint import pprint  # 美观打印字典

# 计算当前选择数据集的绝对路径
data_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", world.dataset)
)

# 实例化数据集加载器
dataset = dataloader.Loader(path=data_path)

# 打印核心配置信息，便于实验复现与检查
print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

# 模型名到模型类的映射表
MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN,
    'ngcf': model.NGCF
}