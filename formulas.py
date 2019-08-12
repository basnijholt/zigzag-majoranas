import scipy.constants
import numpy as np

HBAR = scipy.constants.hbar
HBAR_MEV = scipy.constants.hbar / (scipy.constants.eV * 1e-3)
PI = np.pi
M_EV_NM = scipy.constants.m_e / (scipy.constants.eV * 1e-3) / 1e18


def thouless_energy(vF, W):
    return HBAR_MEV * (PI / 2) * vF / W


def fermi_wavenumber(mu, m_eff):
    return np.sqrt(2 * (m_eff * M_EV_NM) * mu) / HBAR_MEV


def fermi_velocity(mu, m_eff):
    kf = fermi_wavenumber(mu, m_eff)
    return HBAR_MEV * kf / (m_eff * M_EV_NM)


def density_to_mu(density, m_eff):
    return HBAR_MEV ** 2 * PI * density / (m_eff * M_EV_NM)
