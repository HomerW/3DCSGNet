import matplotlib.pyplot as plt
import numpy as np
import h5py
import time

def vis_voxels(voxels, name):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(np.rot90(np.rot90(voxels, 1, (0, 1)), 3, (0, 2)), edgecolor='k')

    # plt.savefig(f"{name}.png")
    # plt.close()
    plt.show()

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

def draw_cube(center, dims, angle, axis):

    axis_dict = {
        "x": [1, 0, 0],
        "y": [0, 1, 0],
        "z": [0, 0, 1]
    }
    rot = getRotMatrix(angle, axis_dict[axis])

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

vis_voxels(draw_cube([32, 32, 16], [24, 24, 8], 75, "x"), "test")
# FILENAME = 'data/03001627_chair/03001627_vox.hdf5'
# data = h5py.File(FILENAME, 'r')
# all_voxels = np.flip(data['voxels'][:], axis=3)
# print(all_voxels.shape)
# vis_voxels(all_voxels[0, :, :, :, 0], "test")
