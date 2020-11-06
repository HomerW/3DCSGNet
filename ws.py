import sys
import numpy as np
import torch
import torch.optim as optim
from torch.autograd.variable import Variable
from src.Utils import read_config
from src.Generator.shapenet_generator import get_shapenet_data
from src.Models.loss import losses_joint
from src.Models.models import CsgNet, ParseModelOutput
from src.Utils.learn_utils import LearningRate
from src.Utils.train_utils import prepare_input_op, Callbacks
import deepdish as dd

config = read_config.Config("config.yml")

config.train_size = 3000
config.test_size = 100

train_dataset, val_dataset = get_shapenet_data('data/03001627_chair/03001627_train_vox.hdf5',
                                               'data/03001627_chair/03001627_val_vox.hdf5',
                                               3000,
                                               100,
                                               batch_size=config.batch_size)

with open("draws.txt", "r") as file:
    unique_draws = file.readlines()
unique_draws = [x.strip() for x in unique_draws]
# primitives = dd.io.load('data/primitives.h5')
max_len = 7

csgnet = CsgNet(grid_shape=[64, 64, 64], dropout=config.dropout,
                     mode=config.mode, timesteps=7,
                     num_draws=len(unique_draws),
                     in_sz=config.input_size,
                     hd_sz=config.hidden_size,
                     stack_len=config.top_k)
weights = torch.load(config.pretrain_modelpath)
new_weights = {}
for k in weights.keys():
    if k.startswith("module"):
        new_weights[k[7:]] = weights[k]
csgnet.load_state_dict(new_weights)
csgnet.cuda()
for param in csgnet.parameters():
    param.requires_grad = True

Target_expressions = []
Predicted_expressions = []
parser = ParseModelOutput(unique_draws, max_len // 2 + 1, max_len, [64, 64, 64], primitives=None)
csgnet.eval()
IOU = {}
total_iou = 0
Rs = 0.0
batch_idx = 0
for batch in train_dataset:
    with torch.no_grad():
        print(f"batch {batch_idx}/{len(train_dataset) // config.batch_size}")
        batch_idx += 1

        outputs = csgnet.test2(data, max_len)

        stack, _, expressions = parser.get_final_canvas(outputs, if_pred_images=True,
                                                  if_just_expressions=False)
        Predicted_expressions += expressions
        target_expressions = parser.labels2exps(labels, k)
        Target_expressions += target_expressions
        # stacks = parser.expression2stack(expressions)
        data_ = data_[-1, :, 0, :, :, :]
        R = np.sum(np.logical_and(stack, data_), (1, 2, 3)) / (np.sum(
            np.logical_or(stack, data_), (1, 2, 3)) + 1)
        Rs += np.sum(R)
total_iou += Rs
IOU[k] = Rs / ((dataset_sizes[k][1] // config.batch_size) * config.batch_size)
print("IOU for {} len program: ".format(k), IOU[k])


total_iou = total_iou / config.test_size
print ("total IOU score: ", total_iou)
results = {"total_iou": total_iou, "iou": IOU}
