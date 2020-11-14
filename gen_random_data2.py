import random
import numpy as np
from src.Models.models import ParseModelOutputGenData, validity, ParseModelOutput
from src.Generator.parser import Parser
import deepdish as dd
from vis_voxels import vis_voxels
from copy import deepcopy

max_ops = 10
max_len = (max_ops * 2) + 1
parser = ParseModelOutput(max_len // 2 + 1, max_len, [64, 64, 64])
with open("draws_cuboids.txt", "r") as file:
    unique_draws = file.readlines()
unique_draws = [x.strip() for x in unique_draws]
other_parser = Parser()
ops = ["+", "-", "*"]

def get_voxels(exp):
    program = other_parser.parse(exp)
    parser.sim.generate_stack(program, start_scratch=False)
    stack = parser.sim.stack_t
    stack = np.stack(stack, axis=0)[-1, 0, :, :]
    return stack

def clear_stack():
    parser.sim.stack_t = []
    parser.sim.stack.clear()
    parser.sim.stack_t.append(parser.sim.stack.get_items())

def good_overlap(obj1, obj2):
    new = np.logical_or(obj1, obj2)
    # combined shape some percent greater than each individual shape
    total = np.sum(new)
    sum_obj1 = np.sum(obj1)
    sum_obj2 = np.sum(obj2)
    percent1 = (total - sum_obj1) / sum_obj1
    percent2 = (total - sum_obj2) / sum_obj2
    # intersection amount is at least some percentage of combined shape
    # no shape completely overlaps
    intersect_percent = np.sum(np.logical_and(obj1, obj2)) / np.sum(new)
    return percent1 >= 0.05 and percent2 >= 0.05 and intersect_percent >= 0.05

def rand_program():
    # true - primitive or sub-tree, false - operation
    q = [True, True, False]
    hier_ind = 1
    program = ""
    inter_prog = ""
    old_voxels = np.zeros((64, 64, 64))
    first = True

    clear_stack()

    while len(q) > 0:
        value = q.pop(0)
        if value:
            if (random.random() < 0.9 and hier_ind < max_ops) or ((not any(q)) and hier_ind < max_ops):
                q = [True, True, False] + q
                hier_ind += 1
            else:
                # rejection sample prims for overlap
                while True:
                    old_stack_t = deepcopy(parser.sim.stack_t)
                    old_stack = deepcopy(parser.sim.stack)
                    prim = unique_draws[random.choice(range(len(unique_draws)))]
                    new_voxels = get_voxels(inter_prog + prim)
                    if first or good_overlap(old_voxels, new_voxels):
                        program += inter_prog + prim
                        inter_prog = ""
                        old_voxels = new_voxels
                        first = False
                        break
                    parser.sim.stack_t = old_stack_t
                    parser.sim.stack = old_stack
        else:
            ops = ["+", "-", "*"]
            inter_prog += random.choice(ops)

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
    # num_plus = prog.count("+")
    # num_minus = prog.count("-")
    # num_times = prog.count("*")
    # total = num_plus + num_minus + num_times
    # with open(f"{total}_{max_ops}.txt", "a") as file:
    #     file.write(f"{prog}\n")
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
