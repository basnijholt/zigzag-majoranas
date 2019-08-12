from types import SimpleNamespace

import kwant
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

import sns_system


def dispersion(k_x, k_y, params):
    p = SimpleNamespace(**params)
    Ekin = (
        p.hbar ** 2 / (2 * p.m_eff) * (k_x ** 2 + k_y ** 2)
        - p.mu
        + p.m_eff * p.alpha_middle ** 2 / (2 * p.hbar ** 2)
    )
    Erest = np.sqrt(
        p.alpha_middle ** 2 * k_x ** 2
        + (p.alpha_middle * k_y - p.g_factor_middle * p.mu_B * p.B) ** 2
    )
    return Ekin + Erest, Ekin - Erest


def calc_lowest_state(syst_pars_params, syst=None):
    syst_pars, params = syst_pars_params
    if syst is None:
        syst = sns_system.make_zigzag_system(**syst_pars)

    ham = syst.hamiltonian_submatrix(sparse=True, params=params)
    e, ev = mumps_eigsh(ham, 4, 0)
    return np.sort(np.abs(e))[::2]


def mumps_eigsh(matrix, k, sigma, **kwargs):
    """Call sla.eigsh with mumps support.

    Please see scipy.sparse.linalg.eigsh for documentation.
    """

    class LuInv(sla.LinearOperator):
        def __init__(self, matrix):
            instance = kwant.linalg.mumps.MUMPSContext()
            instance.analyze(matrix, ordering="pord")
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


def calc_dos_lowest_state(syst, params, syst_pars):
    """ Calculate density of states for lowest energy
    Parameters
    ----------
    syst : kwant.system.FiniteSystem

    params : dictionary of parameters for syst

    syst_pars: dictionary of system dimensional parameters

    Returns
    -------
    energy : float
        Energy of lowest eigenmode

    energy_gap : float
        Energy gap between first and second mode

    dos : numpy.ndarray
        Density of states in 2d array format
    """
    energies, wfs = calc_spectrum(syst, params, k=6)
    energy_gap = abs(energies[2]) - abs(energies[0])
    wf = sns_system.to_site_ph_spin(syst_pars, wfs[:, 0])
    return abs(energies[0]), energy_gap, np.sum(np.abs(wf) ** 2, axis=2)


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


def bands(lead, params, ks=None):
    if ks is None:
        ks = np.linspace(-3, 3)

    bands = kwant.physics.Bands(lead, params=params)

    if isinstance(ks, (float, int)):
        return bands(ks)
    else:
        return np.array([bands(k) for k in ks])


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
    lim = [0, np.abs(bands(lead, params, ks=0)).min()]
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
    h_k = lead.hamiltonian_submatrix(params=dict(params, mu=0, k_x=k_x), sparse=True)
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
        mus = sla.eigs(chemical_potentials, k=num_bands, sigma=sigma, which="LM")[0]

    real_solutions = abs(np.angle(mus)) < 1e-10

    mus[~real_solutions] = np.nan  # To ensure it returns the same shape vector
    return np.sort(mus.real)


def cell_mats(lead, params, bias=0):
    h = lead.cell_hamiltonian(params=params)
    h -= bias * np.identity(len(h))
    t = lead.inter_cell_hopping(params=params)
    return h, t


def get_h_k(lead, params):
    h, t = cell_mats(lead, params)
    h_k = lambda k: h + t * np.exp(1j * k) + t.T.conj() * np.exp(-1j * k)
    return h_k


def lat_from_syst(syst):
    lats = set(s.family for s in syst.sites)
    if len(lats) > 1:
        raise Exception("No unique lattice in the system.")
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


def make_skew_symmetric(ham):
    """
    Makes a skew symmetric matrix by a matrix multiplication of a unitary
    matrix U. This unitary matrix is taken from the Topology MOOC 0D, but
    that is in a different basis. To get to the right basis one multiplies
    by [[np.eye(2), 0], [0, sigma_y]].

    Parameters:
    -----------
    ham : numpy.ndarray
        Hamiltonian matrix gotten from sys.cell_hamiltonian()

    Returns:
    --------
    skew_ham : numpy.ndarray
        Skew symmetrized Hamiltonian
    """
    W = ham.shape[0] // 4
    I = np.eye(2, dtype=complex)
    sigma_y = np.array([[0, 1j], [-1j, 0]], dtype=complex)
    U_1 = np.bmat([[I, I], [1j * I, -1j * I]])
    U_2 = np.bmat([[I, 0 * I], [0 * I, sigma_y]])
    U = U_1 @ U_2
    U = np.kron(np.eye(W, dtype=complex), U)
    skew_ham = U @ ham @ U.H

    assert is_antisymmetric(skew_ham)

    return skew_ham


def calculate_pfaffian(lead, params):
    """
    Calculates the Pfaffian for the infinite system by computing it at k = 0
    and k = pi.

    Parameters:
    -----------
    lead : kwant.builder.InfiniteSystem object
          The finalized system.

    """
    h_k = get_h_k(lead, params)

    skew_h0 = make_skew_symmetric(h_k(0))
    skew_h_pi = make_skew_symmetric(h_k(np.pi))

    pf_0 = np.sign(pf.pfaffian(1j * skew_h0, sign_only=True).real)
    pf_pi = np.sign(pf.pfaffian(1j * skew_h_pi, sign_only=True).real)
    pfaf = pf_0 * pf_pi

    return pfaf
