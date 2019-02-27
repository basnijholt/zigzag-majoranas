import cmath
from functools import lru_cache
import math
import re
from types import SimpleNamespace

import kwant
from kwant.continuum import discretize
import numpy as np
import scipy.constants
import scipy.interpolate
from scipy.optimize import fsolve
import scipy.sparse as sp
import scipy.sparse.linalg as sla

import peierls
import supercurrent
import supercurrent_matsubara
from shape import *

sigma_0 = np.eye(2)
sigma_z = np.array([[1, 0], [0, -1]])
s0sz = np.kron(sigma_0, sigma_z)

constants = dict(
    # effective mass in kg,
    m_eff=0.023 * scipy.constants.m_e / (scipy.constants.eV * 1e-3) / 1e18,
    hbar=scipy.constants.hbar / (scipy.constants.eV * 1e-3),
    mu_B=scipy.constants.physical_constants['Bohr magneton'][0] / (
        scipy.constants.eV * 1e-3),
    exp=cmath.exp,
    cos=cmath.cos,
    sin=cmath.sin)


def create_parallel_sine(distance, z_x, z_y, rough_edge=None):
    def _parallel_sine(x, distance, z_x, z_y):
        g = lambda t: z_y * math.sin(2*np.pi/z_x*t)
        g_prime = lambda t: z_y * 2*np.pi/z_x*math.cos(2*np.pi/z_x*t)
        def _x(t):
            return t - distance*g_prime(t)/np.sqrt(1 + g_prime(t)**2) - x

        def y(t):
            return g(t) + distance/np.sqrt(1 + g_prime(t)**2)

        t = fsolve(_x, x)
        return y(t)

    xs = np.linspace(0, z_x, 1000)
    ys = [_parallel_sine(x, distance, z_x, z_y) for x in xs]

    if rough_edge is not None:
        X, Y, salt = rough_edge

        # Calculate the unit vector to the sine
        tck = scipy.interpolate.splrep(xs, ys)
        dydx = scipy.interpolate.splev(xs, tck, der=1)
        unit_vectors = 1 / (1 + dydx**2) * np.array([-dydx, np.ones_like(dydx)])

        # Generate a disordered boundary parameterized by (X, Y, salt)
        rand = lambda i: X * kwant.digest.uniform(str(i), salt=str(salt))
        rands = [rand(i) for i in range(int(z_x / Y) - 2)]
        rands = [0, *rands, 0]  # make sure that it starts and ends with 0 (for periodicity)
        ys_disorder = scipy.interpolate.interp1d(
            np.linspace(0, z_x, len(rands)), rands, kind='quadratic')(xs)
        ys_disorder -= ys_disorder.mean()

        # Disorder in the direction normal to the sine
        dxys = ys_disorder * unit_vectors

        # Modify xs and ys to include the disorder
        xs[1:-1] += dxys[0, 1:-1]
        ys[1:-1] += dxys[1, 1:-1]

    parallel_sine = scipy.interpolate.interp1d(xs, ys)

    return lambda x: parallel_sine(x%z_x)


def remove_phs(H):
    return re.sub(r'kron\((sigma_[xyz0]), sigma_[xzy0]\)', r'\1', H)


@lru_cache()
def get_template_strings(
        transverse_soi, mu_from_bottom_of_spin_orbit_bands=True,
        k_x_in_sc=False, with_k_z=False, no_phs=False,
        phs_breaking_potential=False):
    kinetic = "(hbar^2 / (2*m_eff) * (k_y^2 + k_x^2 + k_z^2) - mu {}) * kron(sigma_0, sigma_z)"
    if mu_from_bottom_of_spin_orbit_bands:
        ham_str = kinetic.format("+ m_eff*alpha_middle^2 / (2 * hbar^2)")
    else:
        ham_str = kinetic.format("")

    if not with_k_z:
        ham_str = ham_str.replace('k_z', '0')

    spin_orbit = "- alpha_{} * kron(sigma_y, sigma_z) * k_x"
    ham_normal = ham_str + spin_orbit.format('middle')
    ham_sc_top = ham_str + spin_orbit.format('left')
    ham_sc_bot = ham_str + spin_orbit.format('right')

    if transverse_soi:
        tr_spin_orbit = """+ alpha_{} * kron(sigma_x, sigma_z) * k_y"""
        ham_normal += tr_spin_orbit.format('middle')
        ham_sc_top += tr_spin_orbit.format('left')
        ham_sc_bot += tr_spin_orbit.format('right')

    superconductivity = """+ Delta_{0} * (cos({1}phase / 2) * kron(sigma_0, sigma_x)
                                          + sin({1}phase / 2) * kron(sigma_0, sigma_y))"""
    ham_sc_top += superconductivity.format('left', '-')
    ham_sc_bot += superconductivity.format('right', '+')

    zeeman = "+ g_factor_{} * mu_B * (B_x * kron(sigma_x, sigma_0)"
    zeeman += "+ B_y * kron(sigma_y, sigma_0)"
    zeeman += "+ B_z * kron(sigma_z, sigma_0))"
    ham_normal += zeeman.format('middle')
    ham_sc_top += zeeman.format('left')
    ham_sc_bot += zeeman.format('right')

    ham_barrier = ham_normal + "+ V * kron(sigma_0, sigma_z)"

    if not k_x_in_sc:
        ham_sc_bot = ham_sc_bot.replace('k_x', '0')
        ham_sc_top = ham_sc_top.replace('k_x', '0')

    if phs_breaking_potential:
        ham_normal += "+ V_breaking(x) * kron(sigma_0, sigma_0)"

    template_strings = (('barrier', ham_barrier),
                        ('normal', ham_normal),
                        ('sc_top', ham_sc_top).
                        ('sc_bot', ham_sc_bot))

    if no_phs:
        template_strings = [(k, remove_phs(v)) for k, v in template_strings]

    return template_strings


def get_shapes(shape, z_x, z_y, L_m, L_x, rough_edge=None):
    if shape == 'parallel_curve':
        if rough_edge is not None:
            X, Y, salt = rough_edge

        curve = create_parallel_sine(0, z_x, z_y, rough_edge=None)

        _curve_top = create_parallel_sine(L_m//2, z_x, z_y, rough_edge=(X, Y, salt) if rough_edge else None)
        _below_shape = below_curve(_curve_top)

        _curve_bottom = create_parallel_sine(-L_m//2, z_x, z_y, rough_edge=(X, Y, -salt) if rough_edge else None)
        _above_shape = above_curve(_curve_bottom)

        _middle_shape = (_below_shape * _above_shape)[0:L_x, :]
        top_sc_initial_site = (z_x//4, L_m//2+z_y+L_sc_up//2)
        top_sc_shape = _middle_shape.inverse()[0:L_x, :L_sc_up + L_m//2 + z_y]
        bottom_sc_initial_site = (z_x//4, -L_m//2-z_y-L_sc_down//2)
        bottom_sc_shape = _middle_shape.inverse()[0:L_x, -L_sc_down - L_m//2 - z_y:]

    elif shape == 'sawtooth':
        _curve = lambda x: 4*z_y/z_x*(x%(z_x/2)) - z_y if x%z_x < z_x/2 else -4*z_y/z_x*(x%(z_x/2)) + z_y
        curve = lambda x: _curve(x+z_x/4)
        y_offset = L_m / np.cos(np.arctan(4 * z_y / z_x)) if z_y != 0 else L_m

        _below_shape = below_curve(lambda x: curve(x) + y_offset//2)
        _above_shape = above_curve(lambda x: curve(x) - y_offset//2)

        _middle_shape = (_below_shape * _above_shape)[0:L_x, :]
        top_sc_initial_site = (0, y_offset//2+a)
        top_sc_shape = _middle_shape.inverse()[0:L_x, :L_sc_up + y_offset//2 + z_y]
        bottom_sc_initial_site = (0, -y_offset//2-a)
        bottom_sc_shape = _middle_shape.inverse()[0:L_x, -L_sc_down - y_offset//2 - z_y:]

    #------------
    # Define edge
    edge_shape = _middle_shape.edge() * bottom_sc_shape.outer_edge() * top_sc_shape.outer_edge()

    #------------------------
    # Remove edge from middle
    interior_shape = _middle_shape - edge_shape
    interior_initial_site = (a, a)

    return {'sc_top': (top_sc_shape, top_sc_initial_site),
            'sc_bot': (bottom_sc_shape, bottom_sc_initial_site),
            'normal': (interior_shape, interior_initial_site),
            'edge': edge_shape}


@lru_cache()
def make_system(
    L_m, L_x, L_sc_up, L_sc_down, z_x, z_y, a, shape, transverse_soi,
    mu_from_bottom_of_spin_orbit_bands, k_x_in_sc, wraparound, infinite,
    current, ns_junction, sc_leads=False, no_phs=False, rough_edge=None,
    phs_breaking_potential=False):
    if wraparound and not infinite:
        raise ValueError('If you want to use wraparound, infinite must be True.')
    if sc_leads and not infinite or sc_leads and not wraparound:
        raise ValueError('If you want to use sc_leads, infinite and wraparound must be True.')

    template_strings = get_template_strings(
        transverse_soi, mu_from_bottom_of_spin_orbit_bands, 
        k_x_in_sc, False, no_phs, phs_breaking_potential)

    template = {k: discretize(v, coords=('x', 'y'), grid_spacing=a)
        for k, v in template_strings}

    shapes = get_shapes(shape, z_x, z_y, L_m, L_x, rough_edge)

    syst = kwant.Builder(kwant.TranslationalSymmetry([L_x, 0]) if infinite else None)

    for y in np.arange(-L_m - L_sc_down, L_m + L_sc_up, a):
        syst.fill(template['edge'], shapes['edge'], (0, y))

    syst.fill(template['normal'], *shapes['normal'])

    if L_sc_up > 0:
        syst.fill(template['sc_top'], *shapes['sc_top'])

    if L_sc_down > 0:
        syst.fill(template['sc_bot'], *shapes['sc_bot'])
        
    #####################
    ## Finalize system ##
    #####################

    if infinite and wraparound:
        syst = kwant.wraparound.wraparound(syst)
        if sc_leads:
            lead_up = kwant.Builder(kwant.TranslationalSymmetry([L_x, 0], [0, a]))
            lead_down = kwant.Builder(kwant.TranslationalSymmetry([L_x, 0], [0, -a]))
            lead_up = kwant.wraparound.wraparound(lead_up, keep=1)
            lead_down = kwant.wraparound.wraparound(lead_down, keep=1)
            lead_up.fill(template_top_sc, lambda s: 0 <= s.pos[0] < L_x, (0, 0))
            lead_down.fill(template_bottom_sc, lambda s: 0 <= s.pos[0] < L_x, (0, 0))
            syst.attach_lead(lead_up)
            syst.attach_lead(lead_down)

    return syst.finalized()


def mumps_eigsh(matrix, k, sigma, **kwargs):
    """Call sla.eigsh with mumps support.

    Please see scipy.sparse.linalg.eigsh for documentation.
    """
    class LuInv(sla.LinearOperator):

        def __init__(self, matrix):
            instance = kwant.linalg.mumps.MUMPSContext()
            instance.analyze(matrix, ordering='pord')
            instance.factor(matrix)
            self.solve = instance.solve
            sla.LinearOperator.__init__(self, matrix.dtype, matrix.shape)

        def _matvec(self, x):
            return self.solve(x.astype(self.dtype))

    opinv = LuInv(matrix - sigma * sp.identity(matrix.shape[0]))
    return sla.eigsh(matrix, k, sigma=sigma, OPinv=opinv, **kwargs)


def calc_spectrum(syst, params, k=20):
    ham = syst.hamiltonian_submatrix(params=params, sparse=True)
    (energies, wfs) = mumps_eigsh(ham, k=k, sigma=0)
    return (energies, wfs)


def translation_ev(h, t, tol=1e6):
    """Compute the eigenvalues of the translation operator of a lead.

    Adapted from kwant.physics.leads.modes.

    Parameters
    ----------
    h : numpy array, real or complex, shape (N, N) The unit cell
        Hamiltonian of the lead unit cell.
    t : numpy array, real or complex, shape (N, M)
        The hopping matrix from a lead cell to the one on which self-energy
        has to be calculated (and any other hopping in the same direction).
    tol : float
        Numbers and differences are considered zero when they are smaller
        than `tol` times the machine precision.

    Returns
    -------
    ev : numpy array
        Eigenvalues of the translation operator in the form lambda=r*exp(i*k),
        for |r|=1 they are propagating modes.
    """
    a, b = kwant.physics.leads.setup_linsys(h, t, tol, None).eigenproblem
    ev = kwant.physics.leads.unified_eigenproblem(a, b, tol=tol)[0]
    return ev


def cell_mats(lead, params, bias=0):
    h = lead.cell_hamiltonian(params=params)
    h -= bias * np.identity(len(h))
    t = lead.inter_cell_hopping(params=params)
    return h, t


def gap_minimizer(lead, params, energy):
    """Function that minimizes a function to find the band gap.
    This objective function checks if there are progagating modes at a
    certain energy. Returns zero if there is a propagating mode.

    Parameters
    ----------
    lead : kwant.builder.InfiniteSystem object
        The finalized infinite system.
    params : dict
        A dict that is used to store Hamiltonian parameters.
    energy : float
        Energy at which this function checks for propagating modes.

    Returns
    -------
    minimized_scalar : float
        Value that is zero when there is a propagating mode.
    """
    h, t = cell_mats(lead, params, bias=energy)
    ev = translation_ev(h, t)
    norm = (ev * ev.conj()).real
    return np.min(np.abs(norm - 1))


def find_gap_of_lead(lead, params, tol=1e-6):
    """Finds the gapsize by peforming a binary search of the modes with a
    tolarance of tol.

    Parameters
    ----------
    lead : kwant.builder.InfiniteSystem object
        The finalized infinite system.
    params : dict
        A dict that is used to store Hamiltonian parameters.
    tol : float
        The precision of the binary search.

    Returns
    -------
    gap : float
        Size of the gap.

    Notes
    -----
    For use with `lead = funcs.make_lead()`.
    """
    Es = kwant.physics.Bands(lead, params=params)(k=0)
    lim = [0, np.abs(Es).min()]
    if gap_minimizer(lead, params, energy=0) < 1e-15:
        # No band gap
        gap = 0
    else:
        while lim[1] - lim[0] > tol:
            energy = sum(lim) / 2
            par = gap_minimizer(lead, params, energy)
            if par < 1e-10:
                lim[1] = energy
            else:
                lim[0] = energy
        gap = sum(lim) / 2
    return gap


def phase_bounds_operator(lead, params, k_x=0):
    h_k = lead.hamiltonian_submatrix(params=dict(params, mu=0, k_x=k_x),
        sparse=True)
    sigma_z = sp.csc_matrix(np.array([[1, 0], [0, -1]]))
    _operator = sp.kron(sp.eye(h_k.shape[0] // 2), sigma_z) @ h_k
    return _operator


def find_phase_bounds(lead, params, k_x=0, num_bands=20, sigma=0):
    """Find the phase boundaries.
    Solve an eigenproblem that finds values of chemical potential at which the
    gap closes at momentum k=0. We are looking for all real solutions of the
    form H*psi=0 so we solve sigma_0 * tau_z H * psi = mu * psi.

    Parameters
    -----------
    lead : kwant.builder.InfiniteSystem object
        The finalized infinite system.
    params : dict
        A dictionary that is used to store Hamiltonian parameters.
    k_x : float
        Momentum value, by default set to 0.

    Returns
    --------
    chemical_potential : numpy array
        Twenty values of chemical potential at which a bandgap closes at k=0.
    """
    chemical_potentials = phase_bounds_operator(lead, params, k_x)

    if num_bands is None:
        mus = np.linalg.eigvals(chemical_potentials.todense())
    else:
        mus = sla.eigs(chemical_potentials, k=num_bands, sigma=sigma, which='LM')[0]

    real_solutions = abs(np.angle(mus)) < 1e-10

    mus[~real_solutions] = np.nan # To ensure it returns the same shape vector
    return np.sort(mus.real)


def lat_from_syst(syst):
    lats = set(s.family for s in syst.sites)
    if len(lats) > 1:
        raise Exception('No unique lattice in the system.')
    return list(lats)[0]


def slowest_evan_mode(lead, params):
    """Find the slowest decaying (evanescent) mode.

    It uses an adapted version of the function kwant.physics.leads.modes,
    in such a way that it returns the eigenvalues of the translation operator
    (lamdba = |r|*e^ik). The imaginary part of the wavevector k, is the part
    that makes it decay. The inverse of this Im(k) is the size of a Majorana
    bound state. The norm of the eigenvalue that is closest to one is the
    slowest decaying mode. Also called decay length.

    Parameters:
    -----------
    lead : kwant.builder.InfiniteSystem object
        The finalized infinite system.

    Returns:
    --------
    majorana_length : float
        The length of the Majorana.
    """
    h, t = cell_mats(lead, params, bias=0)
    ev = translation_ev(h, t)
    norm = ev * ev.conj()
    idx = np.abs(norm - 1).argmin()
    a = lat_from_syst(lead).prim_vecs[0, 0]
    majorana_length = np.abs(a / np.log(ev[idx]).real)
    return majorana_length

