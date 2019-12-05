import numpy as np
import torch
import torch.nn as nn


class neural_flow(nn.Module):
    def __init__(self, latent_distribution):
        super().__init__()
        self.network, self.z0 = self.initialize_network()
        self.latent_dist = latent_distribution

    def initialize_network(self):
        # Builds the network and the initial condition parameter for the integration.
        network = nn.Sequential(nn.Linear(2, 30), nn.Tanh(),
                                nn.Linear(30, 30), nn.Tanh(),
                                nn.Linear(30, 30), nn.Tanh(),
                                nn.Linear(30, 1))

        z0 = nn.Sequential(nn.Linear(1, 100), nn.Tanh(),
                           nn.Linear(100, 1))

        return network, z0

    def forward(self, dataset):
        # calculates all the required probabilities
        log_jacob = self.network(dataset.grid_data)#[dataset.unrand_idx, :] #unrandomize output

        dzdx = torch.exp(log_jacob)
        z = self.integrate(dzdx, dataset, self.z0)

        log_pz = self.latent_dist.log_pz(z, dataset.grid_data)#[dataset.unrand_idx, 0:1])
        log_px = log_pz + log_jacob

        return log_px, log_pz, log_jacob, z

    def integrate(self, f, dataset, offset_function):
        # performs integration of jacobian
        dzdx = f.reshape(dataset.grid_dims)
        #x = dataset.grid_data[dataset.unrand_idx, 1:2].reshape(dataset.grid_dims)
        x = dataset.grid_data[:, 1:2].reshape(dataset.grid_dims)

        dx = x[:, 1:] - x[:, :-1]
        integrated = torch.cumsum((dzdx[:, 1:] + dzdx[:, :-1])/2.0 * dx, dim=1)

        z_2D = torch.cat((torch.zeros(dataset.grid_dims[0], 1), integrated), dim=1)
        z = z_2D.reshape(-1, 1) + self.z0(dataset.grid_data[:, 0:1])#[dataset.unrand_idx, :]

        return z

    def sample_grid(self, results_regular_grid, dataset):
        # Samples location of particles from log_px grid.
        t_min, x_min = torch.min(dataset.grid_data, dim=0)[0]
        t_max, x_max = torch.max(dataset.grid_data, dim=0)[0]

        normalized_particle_x = (dataset.particle_x - x_min)/(x_max - x_min) * 2 - 1
        if t_max == t_min: #If we only have one frame
            normalized_particle_t = dataset.particle_t
        else:
            normalized_particle_t = (dataset.particle_t - t_min)/(t_max - t_min) * 2 - 1
        normalized_loc = torch.stack((normalized_particle_x, normalized_particle_t), dim=-1)[None, :, :, :]

        high_D_results = results_regular_grid.reshape(dataset.grid_dims)[None, None, :, :]
        interpolated_data = torch.nn.functional.grid_sample(high_D_results, normalized_loc).squeeze()

        return interpolated_data

    def train(self, dataset, iterations):
        # trains normalizing flow
        optimizer = torch.optim.Adam(self.parameters())

        for it in np.arange(iterations):
            log_px_grid = self.forward(dataset)[0]
            log_px_samples = self.sample_grid(log_px_grid, dataset)
            loss = -torch.mean(log_px_samples)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if it % 1000 == 0:
                print(it, loss.item())

    def sample(self, dataset):
        log_px, log_pz, log_jacob, z = self.forward(dataset)
        px = torch.exp(log_px).cpu().detach().reshape(dataset.grid_dims).numpy()
        pz = torch.exp(log_pz).cpu().detach().reshape(dataset.grid_dims).numpy()
        jacob = torch.exp(log_jacob).cpu().detach().reshape(dataset.grid_dims).numpy()
        z = z.cpu().detach().reshape(dataset.grid_dims).numpy()

        return px, pz, jacob, z
    
    def derivatives(self, log_jacob, z, dataset):
        f = torch.exp(log_jacob)

        df = torch.autograd.grad(f, dataset.grid_data, torch.ones_like(f), create_graph=True)[0]
        f_t = df[:, 0:1]
        f_x = df[:, 1:2]
        f_xx = torch.autograd.grad(f_x, dataset.grid_data, torch.ones_like(f_x), retain_graph=True)[0][:, 1:2]

        z0 = self.z0(dataset.grid_data[:, 0:1])
        z0_t = torch.autograd.grad(z0, dataset.grid_data, torch.ones_like(z0), retain_graph=True)[0][:, 0:1]

        z_t = self.integrate(f_t, dataset, self.z0) - z0 + z0_t

        pz = self.latent_dist.pz(z, 0)
        pz_t = self.latent_dist.pz_t(z, 0)
        pz_z = self.latent_dist.pz_z(z, 0)
        pz_zz = self.latent_dist.pz_zz(z, 0)
        
        # Actually calculating the derivs
        px_t = (pz_t + pz_z * z_t) * f + pz * f_t
        #px_x = pz_z * f**2 + pz * f_x
        px_xx = pz_zz * f**3 + 3 * f * f_x * pz_z + pz * f_xx
        
        return px_t, px_xx
    
    def log_derivs(self, log_jacob, z, dataset):
        f = log_jacob

        df = torch.autograd.grad(f, dataset.grid_data, torch.ones_like(f), create_graph=True)[0]
        f_t = df[:, 0:1]
        f_x = df[:, 1:2]
        f_xx = torch.autograd.grad(f_x, dataset.grid_data, torch.ones_like(f_x), retain_graph=True)[0][:, 1:2]

        z0 = self.z0(dataset.grid_data[:, 0:1])
        z0_t = torch.autograd.grad(z0, dataset.grid_data, torch.ones_like(z0), retain_graph=True)[0][:, 0:1]

        z_t = self.integrate(f_t*torch.exp(f), dataset, 0) - z0 +z0_t
        
        log_px_t = -z * z_t + f_t
        log_px_x = -z * torch.exp(f) + f_x
        log_px_xx = f_xx - torch.exp(f)*(torch.exp(f) + z * f_x)
        
        return log_px_t, log_px_x, log_px_xx