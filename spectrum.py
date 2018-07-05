import kwant
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import sns_system

def sparse_diag(matrix, k, sigma, **kwargs):
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
    (energies, wfs) = sparse_diag(ham, k=k, sigma=0)
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
    (energies, wfs) = calc_spectrum(syst, params, k=6)
    energy_gap = abs(energies[2]) - abs(energies[0])
    wf = sns_system.to_site_ph_spin(syst_pars, wfs[:,0])
    return (abs(energies[0]), energy_gap, np.sum(np.abs(wf)**2, axis=2))

def find_gap(lead, params, energy_precision):
    def has_bands(energy):
        smds = lead.modes(energy=energy, params=params)[1]
        return smds.nmodes>0
    
    energy_ub = params['Delta']
    energy_lb = 0
    
    ub_has_bands = has_bands(energy_ub)
    lb_has_bands = has_bands(energy_lb)

    while(energy_ub - energy_lb > energy_precision):
        energy_middle = (energy_ub + energy_lb) / 2
        if has_bands(energy_middle):
            energy_ub = energy_middle
        else:
            energy_lb = energy_middle
    return energy_lb