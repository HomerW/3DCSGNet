import h5py
import numpy as np
import torch

FILENAME = 'data/03001627_chair/03001627_vox.hdf5'

V_DIM =	64
voxel_inds = ((np.indices((V_DIM, V_DIM, V_DIM)).T + .5) / (V_DIM//2)) -1.
flat_voxel_inds = torch.from_numpy(voxel_inds.reshape(-1, 3)).float()

def writeSPC(pc, fn):
    with open(fn, 'w') as f:
        for a,b,c in pc:
            f.write(f'v {a} {b} {c} \n')

def vis_voxels(voxels, name):
    pos_inds = voxels[:,:,:,0].T.reshape(-1).nonzero().squeeze()
    pos_pts = flat_voxel_inds[pos_inds]
    writeSPC(pos_pts, name)

data = h5py.File(FILENAME, 'r')
all_voxels = torch.from_numpy(data['voxels'][:]).flip(dims=[3])
for i in range(5):
    vis_voxels(all_voxels[i], f'voxel_{i}.obj')
