import world
import dataloader
import model
import torch
import Procedure
from parse import parse_args
import os

args = parse_args()
world.config['multicore'] = args.multicore
world.config['test_u_batch_size'] = args.testbatch
world.tensorboard = 0

dataset = dataloader.Loader(path="../data/"+args.dataset)
if args.model == 'lgn':
    Recmodel = model.LightGCN(world.config, dataset)
elif args.model == 'mf':
    Recmodel = model.PureMF(world.config, dataset)

Recmodel = Recmodel.to(world.device)

# Load Weights safely ignoring new parameters not in checkpoint
weight_file = f"./checkpoints/{args.model}-{args.dataset}-{args.layer}-{args.recdim}.pth.tar"

state_dict = torch.load(weight_file, map_location=world.device)
# Filter out new layer weights if strictly loading old checkpoints for test
model_dict = Recmodel.state_dict()
state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
model_dict.update(state_dict)
Recmodel.load_state_dict(model_dict)
print(f"✅ Loaded model weights from {weight_file}")

# Test
Procedure.Test(dataset, Recmodel, 0, None, world.config['multicore'])
