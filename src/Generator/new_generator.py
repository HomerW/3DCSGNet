"""
Generates training and testing data in mini batches.
"""

import deepdish as dd
import numpy as np
from matplotlib import pyplot as plt
from .parser import Parser
from .stack_cuboids import SimulateStack
from torch.nn import functional as F
from itertools import product
import torch
from torch.utils.data import DataLoader

def _col(samples):
    return samples

def get_random_data(data_labels_path, train_size, val_size, train_batch_size, val_batch_size, time_steps):
    with open(data_labels_path) as data_file:
        expressions = data_file.readlines()

    sim = SimulateStack(time_steps // 2 + 1, [64, 64, 64])
    parser = Parser()
    loc_dict = {(x, y, z): i for (i, (x, y, z)) in enumerate(list(product(list(range(8, 64, 8)), repeat=3)))}
    dim_dict = {(x, y, z): i for (i, (x, y, z)) in enumerate(list(product(list(range(4, 36, 4)), repeat=3)))}

    samples = []
    for index, exp in enumerate(expressions[:train_size + val_size]):
        print(f"processed {index}/{train_size + val_size} for size {time_steps}")
        program = parser.parse(exp)
        sim.generate_stack(program)
        stack = sim.stack_t
        stack = np.stack(stack, axis=0).astype(dtype=np.float32)[-1, 0, :, :, :]

        labels_loc = np.zeros((time_steps + 1,), dtype=np.int64)
        labels_dims = np.zeros((time_steps + 1,), dtype=np.int64)
        labels_rot = np.zeros((time_steps + 1,), dtype=np.int64)
        labels_type = np.zeros((time_steps + 1,), dtype=np.int64)
        for j in range(time_steps):
            if program[j]["type"] == "draw":
                labels_loc[j] = loc_dict[tuple([int(x) for x in program[j]["param"][:3]])]
                labels_dims[j] = dim_dict[tuple([int(x) for x in program[j]["param"][3:6]])]
                labels_rot[j] = int(program[j]["param"][6])
                labels_type[j] = 0
            else:
                labels_type[j] = 1
        # stop token
        labels_type[-1] = 2
        labels = (
            labels_loc,
            labels_dims,
            labels_rot,
            labels_type
        )

        samples.append((stack, labels))

    train_dataset = DataLoader(samples[:train_size], batch_size=train_batch_size, shuffle=True, collate_fn = _col)
    val_dataset = DataLoader(samples[train_size:train_size + val_size], batch_size=val_batch_size, shuffle=False, collate_fn = _col)

    return (train_dataset, val_dataset)
