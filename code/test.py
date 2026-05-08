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

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_PATH, "data", args.dataset)
dataset = dataloader.Loader(path=DATA_PATH)
if args.model == 'lgn':
    Recmodel = model.LightGCN(world.config, dataset)
elif args.model == 'mf':
    Recmodel = model.PureMF(world.config, dataset)

Recmodel = Recmodel.to(world.device)

# Load Weights safely ignoring new parameters not in checkpoint
ckpt_arg = None
if hasattr(args, "ckpt"):
    ckpt_arg = args.ckpt
if ckpt_arg:
    if os.path.isabs(ckpt_arg):
        weight_file = ckpt_arg
    else:
        weight_file = os.path.join(world.FILE_PATH, ckpt_arg)
else:
    weight_file = os.path.join(
        world.FILE_PATH,
        f"{args.model}-{args.dataset}-{args.layer}-{args.recdim}.pth.tar",
    )

if not os.path.exists(weight_file):
    available = []
    if os.path.isdir(world.FILE_PATH):
        available = sorted(
            name for name in os.listdir(world.FILE_PATH) if name.endswith(".pth.tar")
        )
    msg_lines = [f"checkpoint not found: {weight_file}"]
    if available:
        msg_lines.append("available checkpoints:")
        msg_lines.extend([f"  - {name}" for name in available])
    raise FileNotFoundError("\n".join(msg_lines))

state_dict = torch.load(weight_file, map_location=world.device)
# Filter out new layer weights if strictly loading old checkpoints for test
model_dict = Recmodel.state_dict()
state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
model_dict.update(state_dict)
Recmodel.load_state_dict(model_dict)
print(f"✅ Loaded model weights from {weight_file}")

# Test
Procedure.Test(dataset, Recmodel, 0, None, world.config['multicore'])
