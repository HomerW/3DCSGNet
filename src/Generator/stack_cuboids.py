"""
This constructs stack from the expressions. This is specifically tailored for 3D
CSG. Most of the ideas are taken from our previous work on 2D CSG.
"""

import numpy as np
from .parser import Parser

# Helper function: given angle + normal compute a rotation matrix that will accomplish the operation
def getRotMatrix(angle, normal):
    s = np.sin(angle)
    c = np.cos(angle)

    nx = normal[0]
    ny = normal[1]
    nz = normal[2]

    rotmat = np.stack((
        np.stack((c + (1 - c) * nx * nx, (1 - c) * nx * ny - s * nz, (1 - c) * nx * nz + s * ny)),
        np.stack(((1 - c) * nx * ny + s * nz, c + (1 - c) * ny * ny, (1 - c) * ny * nz - s * nx)),
        np.stack(((1 - c) * nx * nz - s * ny, (1 - c) * ny * nz + s * nx, c + (1 - c) * nz * nz))
    ))
    return rotmat

rot_matrices = [getRotMatrix(-30, [1, 0, 0]), getRotMatrix(0, [1, 0, 0]), getRotMatrix(30, [1, 0, 0])]

class PushDownStack(object):
    """Simple PushDown Stack implements in the form of array"""

    def __init__(self, max_len, canvas_shape):
        _shape = [max_len] + canvas_shape
        self.max_len = max_len
        self.canvas_shape = canvas_shape
        self.items = []
        self.max_len = max_len

    def push(self, item):
        if len(self.items) >= self.max_len:
            assert False, "exceeds max len for stack!!"
        self.items = [item.copy()] + self.items

    def pop(self):
        if len(self.items) == 0:
            assert False, "below min len of stack!!"
        item = self.items[0]
        self.items = self.items[1:]
        return item

    def get_items(self):
        """
        In this we create a fixed shape tensor amenable for further usage
        :return:
        """
        size = [self.max_len] + self.canvas_shape
        stack_elements = np.zeros(size, dtype=bool)
        length = len(self.items)
        for j in range(length):
            stack_elements[j, :, :, :] = self.items[j]
        return stack_elements

    def clear(self):
        """Re-initializes the stack"""
        self.items = []


class SimulateStack:
    """
    Simulates the stack for CSG
    """
    def __init__(self, max_len, canvas_shape):
        """
        :param max_len: max size of stack
        :param canvas_shape: canvas shape
        :param draw_uniques: unique operations (draw + ops)
        """
        self.draw_obj = Draw(canvas_shape=canvas_shape)
        self.draw = {
            "c": self.draw_obj.draw_cube
        }
        self.op = {"*": self._and, "+": self._union, "-": self._diff}
        self.stack = PushDownStack(max_len, canvas_shape)
        self.stack_t = []
        self.stack.clear()
        self.stack_t.append(self.stack.get_items())
        self.parser = Parser()

    def draw_all_primitives(self, draw_uniques):
        """
        Draws all primitives so that we don't have to draw them over and over.
        :param draw_uniques: unique operations (draw + ops)
        :return:
        """
        self.primitives = {}
        for index, value in enumerate(draw_uniques[0:-4]):
            p = self.parser.parse(value)[0]
            which_draw = p["value"]
            if which_draw == "u" or which_draw == "p":
                # draw cube or sphere
                x = int(p["param"][0])
                y = int(p["param"][1])
                z = int(p["param"][2])
                radius = int(p["param"][3])
                layer = self.draw[which_draw]([x, y, z], radius)

            elif which_draw == "y":
                # draw cylinder
                # TODO check if the order is correct.
                x = int(p["param"][0])
                y = int(p["param"][1])
                z = int(p["param"][2])
                radius = int(p["param"][3])
                height = int(p["param"][4])
                layer = self.draw[p["value"]]([x, y, z], radius, height)
            self.primitives[value] = layer
        return self.primitives

    def get_all_primitives(self, primitives):
        """ Get all primitive from outseide class
        :param primitives: dictionary containing pre-rendered shape primitives
        """
        self.primitives = primitives

    def parse(self, expression):
        """
        NOTE: This method generates terminal symbol for an input program expressions.
        :param expression: program expression in postfix notation
        :return program:
        """
        shape_types = ["c"]
        op = ["*", "+", "-"]
        program = []
        for index, value in enumerate(expression):
            if value in shape_types:
                program.append({})
                program[-1]["type"] = "draw"

                # find where the parenthesis closes
                close_paren = expression[index:].index(")") + index
                program[-1]["value"] = expression[index:close_paren + 1]
            elif value in op:
                program.append({})
                program[-1]["type"] = "op"
                program[-1]["value"] = value
            else:
                pass
        return program

    def generate_stack(self, program: list, start_scratch=True, if_primitives=False):
        """
        Executes the program step-by-step and stores all intermediate stack
        states.
        :param if_primitives: if pre-rendered primitives are given.
        :param program: List with each item a program step
        :param start_scratch: whether to start creating stack from scratch or
        stack already exist and we are appending new instructions. With this
        set to False, stack can be started from its previous state.
        """
        # clear old garbage
        if start_scratch:
            self.stack_t = []
            self.stack.clear()
            self.stack_t.append(self.stack.get_items())

        for index, p in enumerate(program):
            if p["type"] == "draw":
                if if_primitives:
                    # fast retrieval of shape primitive
                    layer = self.primitives[p["value"]]
                    self.stack.push(layer)
                    self.stack_t.append(self.stack.get_items())
                    continue

                center = [int(x) for x in p["param"][:3]]
                dims = [int(x) for x in p["param"][3:6]]
                rot = int(p["param"][6])
                layer = self.draw[p["value"]](center, dims, rot)

                self.stack.push(layer)

                # Copy to avoid orver-write
                # self.stack_t.append(self.stack.items.copy())
                self.stack_t.append(self.stack.get_items())
            else:
                # operate
                obj_2 = self.stack.pop()
                obj_1 = self.stack.pop()
                layer = self.op[p["value"]](obj_1, obj_2)
                self.stack.push(layer)
                # Copy to avoid over-write
                # self.stack_t.append(self.stack.items.copy())
                self.stack_t.append(self.stack.get_items())

    def _union(self, obj1, obj2):
        """Union between voxel grids"""
        return np.logical_or(obj1, obj2)

    def _and(self, obj1, obj2):
        """Intersection between voxel grids"""
        return np.logical_and(obj1, obj2)

    def _diff(self, obj1, obj2):
        """Subtraction between voxel grids"""
        return (obj1 * 1. - np.logical_and(obj1, obj2) * 1.).astype(np.bool)

class SimulateStackGenData:
    """
    Simulates the stack for CSG
    """
    def __init__(self, max_len, canvas_shape):
        """
        :param max_len: max size of stack
        :param canvas_shape: canvas shape
        :param draw_uniques: unique operations (draw + ops)
        """
        self.draw_obj = Draw(canvas_shape=canvas_shape)
        self.draw = {
            "c": self.draw_obj.draw_cube
        }
        self.op = {"*": self._and, "+": self._union, "-": self._diff}
        self.stack = PushDownStack(max_len, canvas_shape)
        self.stack_t = []
        self.stack.clear()
        self.stack_t.append(self.stack.get_items())
        self.parser = Parser()

    def draw_all_primitives(self, draw_uniques):
        """
        Draws all primitives so that we don't have to draw them over and over.
        :param draw_uniques: unique operations (draw + ops)
        :return:
        """
        self.primitives = {}
        for index, value in enumerate(draw_uniques[0:-4]):
            p = self.parser.parse(value)[0]
            which_draw = p["value"]
            if which_draw == "u" or which_draw == "p":
                # draw cube or sphere
                x = int(p["param"][0])
                y = int(p["param"][1])
                z = int(p["param"][2])
                radius = int(p["param"][3])
                layer = self.draw[which_draw]([x, y, z], radius)

            elif which_draw == "y":
                # draw cylinder
                # TODO check if the order is correct.
                x = int(p["param"][0])
                y = int(p["param"][1])
                z = int(p["param"][2])
                radius = int(p["param"][3])
                height = int(p["param"][4])
                layer = self.draw[p["value"]]([x, y, z], radius, height)
            self.primitives[value] = layer
        return self.primitives

    def get_all_primitives(self, primitives):
        """ Get all primitive from outseide class
        :param primitives: dictionary containing pre-rendered shape primitives
        """
        self.primitives = primitives

    def parse(self, expression):
        """
        NOTE: This method generates terminal symbol for an input program expressions.
        :param expression: program expression in postfix notation
        :return program:
        """
        shape_types = ["c"]
        op = ["*", "+", "-"]
        program = []
        for index, value in enumerate(expression):
            if value in shape_types:
                program.append({})
                program[-1]["type"] = "draw"

                # find where the parenthesis closes
                close_paren = expression[index:].index(")") + index
                program[-1]["value"] = expression[index:close_paren + 1]
            elif value in op:
                program.append({})
                program[-1]["type"] = "op"
                program[-1]["value"] = value
            else:
                pass
        return program

    def generate_stack(self, program: list, start_scratch=True, if_primitives=False):
        """
        Executes the program step-by-step and stores all intermediate stack
        states.
        :param if_primitives: if pre-rendered primitives are given.
        :param program: List with each item a program step
        :param start_scratch: whether to start creating stack from scratch or
        stack already exist and we are appending new instructions. With this
        set to False, stack can be started from its previous state.
        """
        # clear old garbage
        if start_scratch:
            self.stack_t = []
            self.stack.clear()
            self.stack_t.append(self.stack.get_items())

        for index, p in enumerate(program):
            if p["type"] == "draw":
                if if_primitives:
                    # fast retrieval of shape primitive
                    layer = self.primitives[p["value"]]
                    self.stack.push(layer)
                    self.stack_t.append(self.stack.get_items())
                    continue

                center = [int(x) for x in p["param"][:3]]
                dims = [int(x) for x in p["param"][3:6]]
                rot = int(p["param"][6])
                layer = self.draw[p["value"]](center, dims, rot)

                self.stack.push(layer)

                # Copy to avoid orver-write
                # self.stack_t.append(self.stack.items.copy())
                self.stack_t.append(self.stack.get_items())
            else:
                # operate
                obj_2 = self.stack.pop()
                obj_1 = self.stack.pop()
                layer = self.op[p["value"]](obj_1, obj_2)
                self.stack.push(layer)
                # Copy to avoid over-write
                # self.stack_t.append(self.stack.items.copy())
                self.stack_t.append(self.stack.get_items())

    def _union(self, obj1, obj2):
        """Union between voxel grids"""
        new = np.logical_or(obj1, obj2)
        # combined shape some percent greater than each individual shape
        total = np.sum(new)
        sum_obj1 = np.sum(obj1)
        sum_obj2 = np.sum(obj2)
        percent1 = (total - sum_obj1) / sum_obj1
        percent2 = (total - sum_obj2) / sum_obj2
        assert percent1 >= 0.05 and percent2 >= 0.05, "violating op"

        return new

    def _and(self, obj1, obj2):
        """Intersection between voxel grids"""
        new = np.logical_and(obj1, obj2)
        # intersection amount is at least some percentage of combined shape
        # no shape completely overlaps
        intersect_percent = np.sum(new) / np.sum(self._union(obj1, obj2))
        assert intersect_percent >= 0.05, "violating op"

        return new

    def _diff(self, obj1, obj2):
        """Subtraction between voxel grids"""
        new = (obj1 * 1. - np.logical_and(obj1, obj2) * 1.).astype(np.bool)
        # obj1 changes by some percent
        sum_obj1 = np.sum(obj1)
        percent = (sum_obj1 - np.sum(new)) / sum_obj1
        assert percent >= 0.05 and percent <= 0.95, "violating op"

        return new

class Draw:
    def __init__(self, canvas_shape=[64, 64, 64]):
        """
        Helper Class for drawing the canvases.
        :param canvas_shape: shape of the canvas on which to draw objects
        """
        self.canvas_shape = canvas_shape


    def draw_cube(self, center, dims, angle):
        rot = rot_matrices[angle]

        dims = [x // 2 for x in dims]
        canvas = np.zeros([64, 64, 64]).astype(bool)
        x_min = center[0] - dims[0]
        x_max = center[0] + dims[0]
        y_min = center[1] - dims[1]
        y_max = center[1] + dims[1]
        z_min = center[2] - dims[2]
        z_max = center[2] + dims[2]
        voxel_centers = np.stack(np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max), np.arange(z_min, z_max))).reshape(3, -1).T
        voxel_centers = ((voxel_centers / 63) * 2) - 1
        rot_centers = np.clip((((voxel_centers @ rot.T) + 1) / 2) * 63, 0, 63).astype(int)
        canvas[rot_centers[:, 0], rot_centers[:, 1], rot_centers[:, 2]] = True

        return canvas
