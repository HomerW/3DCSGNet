import random
import numpy as np
from src.Models.models import ParseModelOutputGenData, validity, ParseModelOutput
from src.Generator.parser import Parser
import deepdish as dd
from vis_voxels import vis_voxels
from copy import deepcopy

max_ops = 2
max_len = (max_ops * 2) + 1
parser = ParseModelOutputGenData(max_len // 2 + 1, max_len, [64, 64, 64])
with open("d.txt", "r") as file:
    unique_draws = file.readlines()
unique_draws = [x.strip() for x in unique_draws]
other_parser = Parser()

def get_voxels(exp):

    program = other_parser.parse(exp)
    # for i in range(1, len(program)):
    #     sub_prog = program[:i]
    #     try:
    #         parser.sim.generate_stack(sub_prog)
    #     except Exception as e:
    #         print(e)
    #         return None
    #     stack = parser.sim.stack_t
    #     stack = np.stack(stack, axis=0)[-1, 0, :, :]
    #     vis_voxels(stack, f"{i}")

    try:
        parser.sim.generate_stack(program, start_scratch=False)
    except Exception as e:
        return None
    # stack = parser.sim.stack_t
    # stack = np.stack(stack, axis=0)[-1, 0, :, :]
    # return stack
    return True

def clear_stack():
    parser.sim.stack_t = []
    parser.sim.stack.clear()
    parser.sim.stack_t.append(parser.sim.stack.get_items())

def rand_program():
    # true - primitive or sub-tree, false - operation
    q = [True, True, False]
    hier_ind = 1
    program = ""
    inter_prog = ""

    clear_stack()

    while len(q) > 0:
        value = q.pop(0)
        if value:
            # if (random.random() < 0.9 and hier_ind < max_ops) or ((not any(q)) and hier_ind < max_ops):
            if (random.random() < 0.5 and hier_ind < max_ops):
                q = [True, True, False] + q
                hier_ind += 1
            else:
                prim = unique_draws[random.choice(range(len(unique_draws)))]
                inter_prog += prim
        else:
            ops = ["+", "-", "*"]
            success = False
            while len(ops) > 0:
                old_stack_t = deepcopy(parser.sim.stack_t)
                old_stack = deepcopy(parser.sim.stack)
                if ops[0] == "+":
                    if len(ops) == 3:
                        p = [0.01, 0.33, 0.66]
                    elif len(ops) == 2:
                        p = [0.01, 0.99]
                    else:
                        p = [1]
                else:
                    if len(ops) == 2:
                        p = [0.35, 0.65]
                    else:
                        p = [1]
                op = np.random.choice(ops, p=p)
                ops.remove(op)
                if get_voxels(inter_prog + op) is not None:
                    success = True
                    program += inter_prog + op
                    inter_prog = ""
                    break
                parser.sim.stack_t = old_stack_t
                parser.sim.stack = old_stack

            if not success:
                return None
            # op = np.random.choice(ops, p=[0.05, 0.275, 0.675])
            # program += op
            # if get_voxels(program) is None:
            #     return None

    # print(np.sum(get_voxels(program)) / (64 ** 3))
    # if not np.sum(get_voxels(program)) > ((64 ** 3) / 4):
    #     return None

    return program

# with open("10.txt", "r") as file:
#     expressions = file.readlines()
# expressions = [x.strip() for x in expressions]
# for n in range(10):
#     program = parser.sim.parse(expressions[n])
#     parser.sim.generate_stack(program, if_primitives=True)
#     stack = parser.sim.stack_t
#     stack = np.stack(stack, axis=0)[-1, 0, :, :]
#     vis_voxels(stack, n)

while True:
    prog = rand_program()
    while prog is None:
        prog = rand_program()
    num_plus = prog.count("+")
    num_minus = prog.count("-")
    num_times = prog.count("*")
    total = num_plus + num_minus + num_times
    with open(f"{total}.txt", "a") as file:
        file.write(f"{prog}\n")
    print(prog)

# with open("10_10.txt", "r") as file:
#     expressions = file.readlines()
# expressions = [x.strip() for x in expressions]
# get_voxels(expressions[0])
#
# num_plus = all_prog.count("+")
# num_minus = all_prog.count("-")
# num_times = all_prog.count("*")
# total = num_plus + num_minus + num_times
# print(num_plus / total)
# print(num_minus / total)
# print(num_times / total)
# with open("data/three_ops/expressions.txt", "r") as file:
#     expressions = file.readlines()
# expressions = [x.strip() for x in expressions]
# np.random.shuffle(expressions)
# for n in range(10):
#     voxels = get_voxels(expressions[n])
#     if voxels is None:
#         print(voxels)
