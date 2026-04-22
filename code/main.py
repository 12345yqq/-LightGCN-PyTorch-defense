import world  # 全局配置（设备、超参数、路径等）
import utils  # 工具函数（损失封装、随机种子等）
from world import cprint  # 彩色日志打印
import torch  # PyTorch 主库
import numpy as np  # 数值计算库（本文件中基本未直接使用）
from tensorboardX import SummaryWriter  # TensorBoard 日志写入器
import time  # 时间工具
import Procedure  # 训练与测试流程模块
from os.path import join  # 路径拼接

# ============================== 固定随机种子，确保结果可复现 ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ============================== 固定随机种子结束 ==============================

import register  # 注册数据集与模型
from register import dataset  # 当前选择的数据集对象

# 根据参数中模型名称，创建模型实例
Recmodel = register.MODELS[world.model_name](world.config, dataset)
# 将模型移动到指定设备（GPU/CPU）
Recmodel = Recmodel.to(world.device)
# 创建 BPR 训练器（内部包含优化器）
bpr = utils.BPRLoss(Recmodel, world.config)

# 获取当前实验对应的权重文件名
weight_file = utils.getFileName()
print(f"load and save to {weight_file}")

# 若指定 --load，则尝试加载已有权重继续训练
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=world.device))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

# 每个正样本对应的负样本数量（当前流程固定为 1）
Neg_k = 1

# 初始化 TensorBoard
if world.tensorboard:
    # 日志目录中拼接时间戳和实验备注
    w: SummaryWriter = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
    )
else:
    # 不启用 TensorBoard 时写入器置空
    w = None
    world.cprint("not enable tensorflowboard")

try:
    # 主训练循环
    for epoch in range(world.TRAIN_epochs):
        # 记录当前轮开始时间（可用于扩展统计）
        start = time.time()

        # 按测试间隔执行评估
        if epoch % world.TEST_INTERVAL == 0:
            cprint("[TEST]")
            Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])

        # 执行一轮 BPR 训练
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
        print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}')

        # 按保存间隔保存模型
        if (epoch + 1) % world.SAVE_INTERVAL == 0:
            torch.save(Recmodel.state_dict(), weight_file)

    # 训练结束后再保存一次最终权重
    torch.save(Recmodel.state_dict(), weight_file)
finally:
    # 无论中途是否异常，最后都安全关闭 TensorBoard
    if world.tensorboard:
        w.close()