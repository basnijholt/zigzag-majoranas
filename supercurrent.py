from functools import partial
import numbers
import warnings

import adaptive
import numpy as np
import kwant

import sns_system
import spectrum


sigma_0 = np.eye(2)
sigma_z = np.array([[1, 0], [0, -1]])
sigz = np.kron(sigma_0, sigma_z)


def fermi_dirac(e, params):
    if params['T'] > 0:
        beta = 1 / (params['k'] * params['T'])
    else:
        return np.double(e <= 0)
    minbetae = np.minimum(100, -beta * e)
    return np.exp(minbetae) / (1 + np.exp(minbetae))


def wrapped_current(
        syst_pars, params, tol=0.01, syst_wrapped=None, transverse_soi=True,
        mu_from_bottom_of_spin_orbit_bands=True, zero_current=1e-15,
        max_points=1e20):
    if syst_wrapped is None:
        syst_wrapped = sns_system.make_wrapped_system(
            **syst_pars, transverse_soi=transverse_soi,
            mu_from_bottom_of_spin_orbit_bands=mu_from_bottom_of_spin_orbit_bands)

    cut_tag = 1
    direction = 'y'

    cut_sites = get_cuts(syst_wrapped, cut_tag, direction)
    current_operator = kwant.operator.Current(syst_wrapped,
                                              onsite=sigz,
                                              where=cut_sites)

    def f(k):
        p = dict(k_x=k, **params)
        ham = syst_wrapped.hamiltonian_submatrix(params=p)
        local_current_operator = current_operator.bind(params=p)

        (en, evs) = np.linalg.eigh(ham)
        I = sum(fermi_dirac(e.real, p) * local_current_operator(ev)
                for e, ev in zip(en, evs.T))
        return sum(I) * params['e'] / params['hbar'] / syst_pars['a'] / (2*np.pi)

    ef_max = (params['mu'] + params['m_eff'] 
        * params['alpha_middle']**2 / 2 / params['hbar']**2)
    kf = np.sqrt(ef_max * 2 * params['m_eff']
                 ) / params['hbar'] * syst_pars['a']
    kmax = min(1.5 * kf, np.pi)

    learner = adaptive.IntegratorLearner(f, [0, kmax], tol)
    runner = adaptive.runner.simple(
        learner, lambda l: l.done() or l.npoints > max_points or (
            l.npoints > 20 and np.max(
                np.abs(
                    list(
                        learner.done_points.values()))) < zero_current))

    I = 2 * learner.igral
    return I

def wrapped_current_3d(
        syst_pars, params, tol=0.01, syst_wrapped=None, transverse_soi=True,
        mu_from_bottom_of_spin_orbit_bands=True, zero_current=1e-15,
        max_points=1e20):

    cut_tag = 1
    direction = 'y'

    cut_sites = get_cuts(syst_wrapped, cut_tag, direction)
    current_operator = kwant.operator.Current(syst_wrapped,
                                              onsite=sigz,
                                              where=cut_sites)

    def f(k):
        p = dict(k_x=k, **params)
        ham = syst_wrapped.hamiltonian_submatrix(params=p)
        local_current_operator = current_operator.bind(params=p)

        (en, evs) = np.linalg.eigh(ham)
        I = sum(fermi_dirac(e.real, p) * local_current_operator(ev)
                for e, ev in zip(en, evs.T))
        return sum(I) * params['e'] / params['hbar'] / syst_pars['a'] / (2*np.pi) / syst_pars['L_z']

    ef_max = (params['mu'] + params['m_eff'] 
        * params['alpha_middle']**2 / 2 / params['hbar']**2)
    kf = np.sqrt(ef_max * 2 * params['m_eff']
                 ) / params['hbar'] * syst_pars['a']
    kmax = min(1.5 * kf, np.pi)

    learner = adaptive.IntegratorLearner(f, [0, kmax], tol)
    runner = adaptive.runner.simple(
        learner, lambda l: l.done() or l.npoints > max_points or (
            l.npoints > 20 and np.max(
                np.abs(
                    list(
                        learner.done_points.values()))) < zero_current))

    I = 2 * learner.igral
    return I

def get_cuts(syst, ind=0, direction='x'):
    """Get the sites at two postions of the specified cut coordinates.

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem
        The finilized kwant system.
    ind : int
        index of slice to cut, cuts will be returned at ind, and ind+1.
    direction : str
        Cut direction, 'x', 'y', or 'z'.
    """
    direction = 'xyz'.index(direction)
    l_cut = [site for site in syst.sites if site.tag[direction] == ind]
    r_cut = [site for site in syst.sites if site.tag[direction] == ind+1]
    assert len(l_cut) == len(r_cut), "x_left and x_right use site.tag not site.pos!"
    return list(zip(l_cut, r_cut))
