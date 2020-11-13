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
from torch.utils.data import DataLoader
from vis_voxels import vis_voxels

device = torch.device("cuda")

config = read_config.Config("config.yml")
with open("draws.txt", "r") as file:
    unique_draws = file.readlines()
unique_draws = [x.strip() for x in unique_draws]
primitives = dd.io.load('data/primitives.h5')
max_len = 7

def _col(samples):
    return samples

def get_csgnet():
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
    return csgnet


def infer_progams(csgnet, train_dataset, val_dataset):
    parser = ParseModelOutput(unique_draws, max_len // 2 + 1, max_len, [64, 64, 64], primitives=primitives)
    csgnet.eval()
    datasets = [("train", train_dataset), ("val", val_dataset)]
    dataset_expressions = []
    dataset_stacks = []
    dataset_labels = []
    for name, dataset in datasets:
        predicted_expressions = []
        predicted_stacks = []
        predicted_labels = []
        IOU = {}
        total_iou = 0
        Rs = 0.0
        batch_idx = 0
        count = 0
        for batch in dataset:
            with torch.no_grad():
                print(f"batch {batch_idx}/{len(train_dataset)}")
                batch_idx += 1
                count += len(batch)

                vis_voxels(batch.squeeze().numpy()[:5], "gt")

                batch = batch.to(device)

                outputs = csgnet.test2(batch, max_len)

                labels = [torch.max(o, 1)[1].data.cpu().numpy() for o in outputs]
                labels += [np.full((len(batch),), len(unique_draws) - 1)]

                stack, _, expressions = parser.get_final_canvas(outputs, if_pred_images=True,
                                                          if_just_expressions=False)

                vis_voxels(stack[:5], "gen")
                break
                predicted_expressions += expressions
                predicted_stacks.append(stack)
                predicted_labels.append(np.stack(labels).transpose())

                # stacks = parser.expression2stack(expressions)
                data_ = batch.squeeze().cpu().numpy()
                R = np.sum(np.logical_and(stack, data_), (1, 2, 3)) / (np.sum(
                    np.logical_or(stack, data_), (1, 2, 3)) + 1)
                Rs += np.sum(R)
        IOU = Rs / count
        print(f"IOU on ShapeNet {name}: {IOU}")
        dataset_expressions.append(predicted_expressions)
        dataset_stacks.append(np.concatenate(predicted_stacks, axis=0))
        dataset_labels.append(np.concatenate(predicted_labels, axis=0))

    train_samples = list(zip(dataset_labels[0], list(dataset_stacks[0])))
    val_samples = list(zip(dataset_labels[1], list(dataset_stacks[1])))

    train_dataset = DataLoader(train_samples, batch_size=config.batch_size, shuffle=True, collate_fn=_col)
    val_dataset = DataLoader(val_samples, batch_size=config.batch_size, shuffle=False, collate_fn=_col)

    return train_dataset, val_dataset


def train_model(csgnet, train_dataset, val_dataset, max_epochs=None):
    if max_epochs is None:
        epochs = 100
    else:
        epochs = max_epochs

    optimizer = optim.Adam([para for para in csgnet.parameters() if para.requires_grad],
                           weight_decay=config.weight_decay, lr=config.lr)

    reduce_plat = LearningRate(optimizer, init_lr=config.lr, lr_dacay_fact=0.2,
                               lr_decay_epoch=3, patience=config.patience)

    best_state_dict = None
    patience = 3
    prev_test_loss = 1e20
    prev_test_reward = 0
    num_worse = 0
    for epoch in range(100):
        train_loss = 0
        Accuracies = []
        csgnet.train()
        # Number of times to accumulate gradients
        num_accums = config.num_traj
        batch_idx = 0
        count = 0
        for batch in train_dataset:
            labels = np.stack([x[0] for x in batch])
            data = np.stack([x[1] for x in batch])
            if not len(labels) == config.batch_size:
                continue
            optimizer.zero_grad()
            loss_sum = Variable(torch.zeros(1)).cuda().data

            one_hot_labels = prepare_input_op(labels, len(unique_draws))
            one_hot_labels = Variable(torch.from_numpy(one_hot_labels)).cuda()
            data = Variable(torch.from_numpy(data)).cuda().unsqueeze(-1).float()
            labels = Variable(torch.from_numpy(labels)).cuda()

            # forward pass
            outputs = csgnet.forward2([data, one_hot_labels, max_len])

            loss = losses_joint(outputs, labels, time_steps=max_len + 1) / num_accums
            loss.backward()
            loss_sum += loss.data

            batch_idx += 1
            count += len(data)

            if batch_idx % num_accums == 0:
                # Clip the gradient to fixed value to stabilize training.
                torch.nn.utils.clip_grad_norm_(csgnet.parameters(), 20)
                optimizer.step()
                l = loss_sum
                train_loss += l
                # print(f'train loss batch {batch_idx}: {l}')

        mean_train_loss = (train_loss * num_accums) / (count // config.batch_size)
        print(f'train loss epoch {epoch}: {float(mean_train_loss)}')
        del data, loss, loss_sum, train_loss, outputs

        test_losses = 0
        acc = 0
        csgnet.eval()
        test_reward = 0
        batch_idx = 0
        count = 0
        for batch in val_dataset:
            labels = np.stack([x[0] for x in batch])
            data = np.stack([x[1] for x in batch])
            if not len(labels) == config.batch_size:
                continue
            parser = ParseModelOutput(unique_draws,
                                      stack_size=(max_len + 1) // 2 + 1,
                                      steps=max_len,
                                      canvas_shape=[64, 64, 64],
                                      primitives=primitives)
            with torch.no_grad():
                one_hot_labels = prepare_input_op(labels, len(unique_draws))
                one_hot_labels = Variable(torch.from_numpy(one_hot_labels)).cuda()
                data = Variable(torch.from_numpy(data)).cuda().unsqueeze(-1).float()
                labels = Variable(torch.from_numpy(labels)).cuda()

                test_output = csgnet.forward2([data, one_hot_labels, max_len])

                l = losses_joint(test_output, labels, time_steps=max_len + 1).data
                test_losses += l
                acc += float((torch.argmax(torch.stack(test_output), dim=2).permute(1, 0) == labels).float().sum()) \
                / (labels.shape[0] * labels.shape[1])

                test_output = csgnet.test2(data, max_len)

                stack, _, _ = parser.get_final_canvas(test_output, if_pred_images=True,
                                                      if_just_expressions=False)
                data_ = data.squeeze().cpu().numpy()
                R = np.sum(np.logical_and(stack, data_), (1, 2, 3)) / (
                np.sum(np.logical_or(stack, data_), (1, 2, 3)) + 1)
                test_reward += np.sum(R)

            batch_idx += 1
            count += len(data)

        test_reward = test_reward / count

        test_loss = test_losses / (count // config.batch_size)
        acc = acc / (count // config.batch_size)

        if test_loss < prev_test_loss:
            prev_test_loss = test_loss
            best_state_dict = csgnet.state_dict()
            num_worse = 0
        else:
            num_worse += 1
        if num_worse >= patience:
            csgnet.load_state_dict(best_state_dict)
            break

        print(f'test loss epoch {epoch}: {float(test_loss)}')
        print(f'test IOU epoch {epoch}: {test_reward}')
        print(f'test acc epoch {epoch}: {acc}')
        if config.if_schedule:
            reduce_plat.reduce_on_plateu(-test_reward)

        del test_losses, test_output
        if test_reward > prev_test_reward:
            prev_test_reward = test_reward

def run(iterations):
    config.train_size = 66
    config.test_size = 66
    train_dataset, val_dataset = get_shapenet_data('data/03001627_chair/03001627_train_vox.hdf5',
                                                   'data/03001627_chair/03001627_val_vox.hdf5',
                                                   config.train_size,
                                                   config.test_size,
                                                   batch_size=config.batch_size)

    csgnet = get_csgnet()
    for iter in range(iterations):
        inferred_train, inferred_val = infer_progams(csgnet, train_dataset, val_dataset)
        # train_model(csgnet, inferred_train, inferred_val)
        #
        # torch.save(csgnet.state_dict(), "trained_models/ws.pt")

run(1)
