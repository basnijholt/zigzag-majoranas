import numpy as np
import kwant
import sns_system
import spectrum
import numbers
import warnings
from functools import partial
import adaptive


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

    cut_tag = 0
    direction = 1

    (cut_indices, cut_sites) = get_cut_sites_and_indices(
        syst_wrapped, cut_tag, direction)
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
        return sum(I) * params['e'] / params['hbar']

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

    I = 2 * learner.igral / syst_pars['a']
    return I


def get_cut_sites_and_indices(syst, cut_tag, direction):
    l_cut = []
    r_cut = []
    cut_indices = []

    for site_idx, site in enumerate(syst.sites):
        if site.tag[direction] == cut_tag:
            l_cut.append(site)
            temp = [
                4 * site_idx,
                4 * site_idx + 1,
                4 * site_idx + 2,
                4 * site_idx + 3]
            cut_indices.append(temp)
        if site.tag[direction] == cut_tag + 1:
            r_cut.append(site)
            temp = [
                4 * site_idx,
                4 * site_idx + 1,
                4 * site_idx + 2,
                4 * site_idx + 3]
            cut_indices.append(temp)

    cut_indices = np.hstack(cut_indices)
    cut_sites = list(zip(l_cut, r_cut))

    return (cut_indices, cut_sites)
