"""
Train the network using mixture of programs.
"""
import sys
import numpy as np
import torch
import torch.optim as optim
from torch.autograd.variable import Variable
from src.Utils import read_config
from src.Generator.generator import Generator
# from src.Generator.new_generator import get_random_data
from src.Models.loss import losses_joint
from src.Models.models import CsgNet, ParseModelOutput
from src.Utils.learn_utils import LearningRate
from src.Utils.train_utils import prepare_input_op, Callbacks
import deepdish as dd
import time
import torch.nn.functional as F

if len(sys.argv) > 1:
    config = read_config.Config(sys.argv[1])
else:
    config = read_config.Config("config.yml")

model_name = config.model_path.format(config.proportion,
                                      config.top_k,
                                      config.hidden_size,
                                      config.batch_size,
                                      config.optim, config.lr,
                                      config.weight_decay,
                                      config.dropout,
                                      "mix",
                                      config.mode)
print(config.config)

config.write_config("log/configs/{}_config.json".format(model_name))


callback = Callbacks(config.batch_size, "log/db/{}".format(model_name))
callback.add_element(["train_loss", "test_loss", "train_mse", "test_mse"])

# data_labels_paths = {3: "data/one_op/expressions.txt",
#                      5: "data/two_ops/expressions.txt",
#                      7: "data/three_ops/expressions.txt"}
data_labels_paths = {(k*2)+1: f"data/new_synthetic/{k}.txt" for k in range(1, 11)}

proportion = config.proportion  # proportion is in percentage. vary from [1, 100].

# First is training size and second is validation size per program length
# dataset_sizes = {3: [proportion * 1000, proportion * 250],
#                  5: [proportion * 2000, proportion * 500],
#                  7: [proportion * 4000, proportion * 100]}
# dataset_sizes = {(k*2)+1: [max(20000, int((k / 2))*10000), 100] for k in range(1, 11)}
dataset_sizes = {(k*2)+1: [3000, 120] for k in range(1, 11)}

config.train_size = sum(dataset_sizes[k][0] for k in dataset_sizes.keys())
config.test_size = sum(dataset_sizes[k][1] for k in dataset_sizes.keys())
types_prog = len(dataset_sizes)

# primitives = dd.io.load('data/primitives_cuboids.h5')
primitives = None
generator = Generator(data_labels_paths=data_labels_paths,
                      batch_size=config.batch_size,
                      time_steps=max(data_labels_paths.keys()),
                      stack_size=max(data_labels_paths.keys()) // 2 + 1,
                      primitives = primitives)

imitate_net = CsgNet(grid_shape=[64, 64, 64], dropout=config.dropout,
                     mode=config.mode, timesteps=max(data_labels_paths.keys()),
                     in_sz=config.input_size,
                     hd_sz=config.hidden_size,
                     stack_len=config.top_k)

# If you want to use multiple GPUs for training.
cuda_devices = torch.cuda.device_count()
if torch.cuda.device_count() > 1:
    imitate_net.cuda_devices = torch.cuda.device_count()
    print("using multi gpus", flush=True)
    imitate_net = torch.nn.DataParallel(imitate_net, device_ids=[0, 1], dim=0)
imitate_net.cuda()

# if config.preload_model:
#     imitate_net.load_state_dict(torch.load(config.pretrain_modelpath))

for param in imitate_net.parameters():
    param.requires_grad = True

if config.optim == "sgd":
    optimizer = optim.SGD(
        [para for para in imitate_net.parameters() if para.requires_grad],
        weight_decay=config.weight_decay,
        momentum=0.9, lr=config.lr, nesterov=False)

elif config.optim == "adam":
    optimizer = optim.Adam(
        [para for para in imitate_net.parameters() if para.requires_grad],
        weight_decay=config.weight_decay, lr=config.lr)

reduce_plat = LearningRate(optimizer, init_lr=config.lr, lr_dacay_fact=0.2,
                            lr_decay_epoch=3, patience=config.patience)

train_gen_objs = {}
test_gen_objs = {}
# gen_objs = {}

# Prefetching minibatches
for k in data_labels_paths.keys():
    # if using multi gpu training, train and test batch size should be multiple of
    # number of GPU edvices.
    train_batch_size = config.batch_size // types_prog
    test_batch_size = config.batch_size // types_prog
    train_gen_objs[k] = generator.get_train_data(train_batch_size,
                                                 k,
                                                 num_train_images=dataset_sizes[k][0],
                                                 if_primitives=(not primitives is None),
                                                 if_jitter=False)
    test_gen_objs[k] = generator.get_test_data(test_batch_size,
                                               k,
                                               num_train_images=dataset_sizes[k][0],
                                               num_test_images=dataset_sizes[k][1],
                                               if_primitives=(not primitives is None),
                                               if_jitter=False)
    # gen_objs[k] = get_random_data(data_labels_paths[k],
    #                                     dataset_sizes[k][0],
    #                                     dataset_sizes[k][1],
    #                                     train_batch_size,
    #                                     test_batch_size,
    #                                     k)

def one_hot_labels(labels):
    labels_loc, labels_dims, labels_rot, labels_type = labels
    oh_labels = torch.cat((
        F.one_hot(torch.from_numpy(labels_loc), 343),
        F.one_hot(torch.from_numpy(labels_dims), 512),
        F.one_hot(torch.from_numpy(labels_rot), 3),
        F.one_hot(torch.from_numpy(labels_type), 5)
    ), dim=2).float()
    start_token = np.zeros((oh_labels.shape[0], 1, oh_labels.shape[2]))
    start_token[:, :, -1] = 1
    oh_labels = torch.cat((torch.from_numpy(start_token).float(), oh_labels), dim=1)
    return oh_labels

prev_test_loss = 1e20
prev_test_reward = 0
test_size = config.test_size
batch_size = config.batch_size
for epoch in range(0, config.epochs):
    # gen_objs_iters = {k: (iter(v[0]), iter(v[1])) for k, v in gen_objs.items()}
    start_time = time.time()
    train_loss = 0
    Accuracies = []
    imitate_net.train()
    # Number of times to accumulate gradients
    num_accums = config.num_traj
    for batch_idx in range(config.train_size // (config.batch_size * config.num_traj)):
        batch_start = time.time()
        optimizer.zero_grad()
        loss_sum = Variable(torch.zeros(1)).cuda().data
        for _ in range(num_accums):
            for k in data_labels_paths.keys():
                # samples = next(gen_objs_iters[k][0])
                # data = np.stack([x[0] for x in samples])
                # labels = (np.stack([x[1][0] for x in samples]),
                #           np.stack([x[1][1] for x in samples]),
                #           np.stack([x[1][2] for x in samples]),
                #           np.stack([x[1][3] for x in samples]))
                data, labels = next(train_gen_objs[k])
                oh_labels = one_hot_labels(labels).cuda()

                # data = data[:, :, 0:config.top_k + 1, :, :, :]
                data = Variable(torch.from_numpy(data)).cuda()

                # forward pass
                outputs = imitate_net([data, oh_labels, k])

                loss = imitate_net.loss_function(outputs, labels)
                loss.backward()
                loss_sum += loss.data
        batch_end = time.time()
        print(f"batch time: {batch_end - batch_start}")

        # Clip the gradient to fixed value to stabilize training.
        torch.nn.utils.clip_grad_norm_(imitate_net.parameters(), 20)
        optimizer.step()
        l = loss_sum
        train_loss += l
        print('train_loss_batch', l.cpu().numpy(), epoch * (
            config.train_size //
            (config.batch_size * num_accums)) + batch_idx)
    mean_train_loss = train_loss / (config.train_size // (config.batch_size * num_accums)) / types_prog
    print('train_loss', mean_train_loss.cpu().numpy(), epoch)
    del data, loss, loss_sum, train_loss, outputs

    end_time = time.time()
    print(f"TIME: {end_time - start_time}")

    test_losses = 0
    imitate_net.eval()
    test_reward = 0
    num_correct = 0
    for batch_idx in range(config.test_size // config.batch_size):
        for k in data_labels_paths.keys():
            with torch.no_grad():
                parser = ParseModelOutput(stack_size=(k + 1) // 2 + 1,
                                          steps=k,
                                          canvas_shape=[64, 64, 64])
                # samples = next(gen_objs_iters[k][1])
                # data_ = np.stack([x[0] for x in samples])
                # labels = (np.stack([x[1][0] for x in samples]),
                #           np.stack([x[1][1] for x in samples]),
                #           np.stack([x[1][2] for x in samples]),
                #           np.stack([x[1][3] for x in samples]))
                data_, labels = next(test_gen_objs[k])
                oh_labels = one_hot_labels(labels).cuda()

                data = Variable(torch.from_numpy(data_)).cuda()

                if cuda_devices > 1:
                    test_output = imitate_net.module.test([data, oh_labels, k])
                else:
                    test_output = imitate_net.test([data, oh_labels, k])

                l = imitate_net.loss_function(test_output, [x[:, :-1] for x in labels])
                test_losses += l

                stack, correct_programs, _ = parser.get_final_canvas(test_output, if_pred_images=True,
                                                      if_just_expressions=False)
                num_correct += len(correct_programs)

                # data_ = data_[-1, :, 0, :, :, :]
                R = np.sum(np.logical_and(stack, data_), (1, 2, 3)) / (
                np.sum(np.logical_or(stack, data_), (1, 2, 3)) + 1)
                test_reward += np.sum(R)

    test_reward = test_reward / (test_size // batch_size) / ((batch_size // types_prog) * types_prog)

    test_loss = test_losses.cpu().numpy() / (config.test_size // config.batch_size) / types_prog
    print('test_loss', test_loss, epoch)
    print('test_IOU', test_reward / (config.test_size // config.batch_size), epoch)
    print('percent correct programs', num_correct / config.test_size, epoch)
    callback.add_value({
        "test_loss": test_loss,
    })
    print ("Average test IOU: {} at {} epoch".format(test_reward, epoch))
    if config.if_schedule:
        reduce_plat.reduce_on_plateu(-test_reward)

    del test_losses, test_output
    if test_reward > prev_test_reward:
        torch.save(imitate_net.state_dict(),
                   "trained_models/new_synthetic.pth")
        prev_test_reward = test_reward
    callback.dump_all()
