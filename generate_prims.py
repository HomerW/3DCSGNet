from itertools import product
import deepdish as dd
import numpy as np


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

def draw_cube(center, dims, angle):

    # axis_dict = {
    #     "x": [1, 0, 0],
    #     "y": [0, 1, 0],
    #     "z": [0, 0, 1]
    # }
    # rot = getRotMatrix(angle, axis_dict[axis])
    rot = rot_matrices[angle]

    dims = [x // 2 for x in dims]
    # canvas = np.zeros([64, 64, 64])
    for x in range(center[0] - dims[0], center[0] + dims[0]):
        for y in range(center[1] - dims[1], center[1] + dims[1]):
            for z in range(center[2] - dims[2], center[2] + dims[2]):
                voxel_center = np.array([((pt / 63) * 2) - 1 for pt in [x, y, z]])
                rot_center = ((((rot @ voxel_center) + 1) / 2) * 63).astype(int)
                if (rot_center > 63).any() or (rot_center < 0).any():
                    return False
                # rx, ry, rz = rot_center
                # canvas[rx, ry, rz] = True

    return True

prims = []
voxels = {}
j = 0
for loc in product(list(range(8, 64, 8)), repeat=3):
    for dims in product(list(range(4, 36, 4)), repeat=3):
        good_params = []
        for i in range(3):
            _min = loc[i] - (dims[i] // 2)
            _max = loc[i] + (dims[i] // 2)
            good_params.append(_min >= 0 and _max <= 64)
        if all(good_params):
            for rot in range(3):
                rot_dict = {
                    0: ["x", -30],
                    1: ["x", 0],
                    2: ["x", 30]
                }
                # inside = draw_cube(loc, dims, rot)
                # if inside:
                #     print(j)
                #     j += 1
                expression = f"c({loc[0]},{loc[1]},{loc[2]},{dims[0]},{dims[1]},{dims[2]},{rot})"
                prims.append(expression)

print(len(prims))

# dd.io.save('primitives_cuboids.h5', voxels)

# with open("draws_cuboids_not_inside.txt", "w") as file:
#     for p in prims:
#         file.write(f"{p}\n")
