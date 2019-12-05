import torch
import numpy as np
from collections import namedtuple


def prepare_data(x_particles, t_particles, x_sample, t_sample):
    t_grid, x_grid = np.meshgrid(t_sample, x_sample, indexing='ij')
    rand_idx = np.random.permutation(t_grid.size)
    grid_data = torch.tensor(np.concatenate((t_grid.reshape(-1, 1), x_grid.reshape(-1, 1)), axis=1), dtype=torch.float32, requires_grad=True)
    grid_dims = x_grid.shape

    particle_x = torch.tensor(x_particles, dtype=torch.float32)
    particle_t = torch.tensor(np.ones_like(x_particles)*t_particles[:, None], dtype=torch.float32)

    unrand_idx = np.empty(rand_idx.size, rand_idx.dtype)
    unrand_idx[rand_idx] = np.arange(rand_idx.size) # used to unrandomize the data

    neural_flow_data = namedtuple('neural_flow_data', ['particle_x', 'particle_t', 'grid_data', 'grid_dims', 'unrand_idx'])
    dataset = neural_flow_data(particle_x, particle_t, grid_data, grid_dims, unrand_idx)

    return dataset
