import numpy as np
import torch


class gaussian:
    @staticmethod
    def log_pz(z, t):
        log_pz = -1/2 * np.log(2*np.pi) - z**2 / 2.0

        return log_pz

    @ staticmethod
    def pz(z, t):
        pz = 1 / np.sqrt(2*np.pi) * torch.exp(-z**2/2)

        return pz

    @staticmethod
    def pz_z(z, t):
        pz_z = -z * gaussian.pz(z, t)

        return pz_z

    @staticmethod
    def pz_zz(z, t):
        pz_zz = (z**2 - 1) * gaussian.pz(z, t)

        return pz_zz

    @staticmethod
    def pz_t(z, t):
        pz_t = torch.zeros_like(z)

        return pz_t
