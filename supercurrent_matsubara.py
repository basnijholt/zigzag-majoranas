#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018, Bas Nijholt, Viacheslav Ostroukh, and Anton Akhmerov
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the TU Delft nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL BAS NIJHOLT BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Functions library used to calculate supercurrents.

import kwant
import numpy as np
import scipy.optimize


def get_cuts(syst, ind=0, direction='x'):
    """Get the sites at two postions of the specified cut coordinates.

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem
        The finilized kwant system.
    ind : int
        index of slice to cut, cuts will be returned at ind and ind+1.
    direction : str
        Cut direction, 'x', 'y', or 'z'.
    """
    direction = 'xyz'.index(direction)
    l_cut = [site for site in syst.sites() if site.tag[direction] == ind]
    r_cut = [site for site in syst.sites() if site.tag[direction] == ind+1]
    assert len(l_cut) == len(r_cut), "x_left and x_right use site.tag not site.pos!"
    return l_cut, r_cut


def add_vlead(syst, norbs, l_cut, r_cut):
    dim = norbs * (len(l_cut) + len(r_cut))
    vlead = kwant.builder.SelfEnergyLead(
        lambda energy, args: np.zeros((dim, dim)), l_cut + r_cut)
    syst.leads.append(vlead)
    return syst


def hopping_between_cuts(syst, r_cut, l_cut, electron_blocks):
    r_cut_sites = [syst.sites.index(site) for site in r_cut]
    l_cut_sites = [syst.sites.index(site) for site in l_cut]

    def hopping(syst, params):
        H = syst.hamiltonian_submatrix(params=params,
                                       to_sites=l_cut_sites,
                                       from_sites=r_cut_sites)
        return electron_blocks(H)
    return hopping


def matsubara_frequency(n, params):
    """n-th fermionic Matsubara frequency at temperature T.

    Parameters
    ----------
    n : int
        n-th Matsubara frequency

    Returns
    -------
    float
        Imaginary energy.
    """
    return (2*n + 1) * np.pi * params['k'] * params['T'] * 1j


def null_H(syst, params, n, electron_blocks):
    """Return the Hamiltonian (inverse of the Green's function) of
    the electron part at zero phase.

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem
        The finilized kwant system.
    params : dict
        A container that is used to store Hamiltonian parameters.
    n : int
        n-th Matsubara frequency
    electron_blocks: function
        A function that takes the electron blocks from the Hamiltonian.

    Returns
    -------
    numpy.array
        The Hamiltonian at zero energy and zero phase.
    """
    assert isinstance(syst.leads[0], kwant.builder.SelfEnergyLead)
    en = matsubara_frequency(n, params)
    gf = kwant.greens_function(syst, en, out_leads=[0], in_leads=[0],
                               check_hermiticity=False, params=params)
    return np.linalg.inv(electron_blocks(gf.data))


def gf_from_H_0(H_0, t):
    """Returns the Green's function at a phase that is defined inside `t`.
    See doc-string of `current_from_H_0`.
    """
    H = np.copy(H_0)
    dim = t.shape[0]
    H[:dim, dim:] -= t.T.conj()
    H[dim:, :dim] -= t
    return np.linalg.inv(H)


def current_from_H_0(H_0_cache, H12, phase, params):
    """Uses Dyson’s equation to obtain the Hamiltonian for other
    values of `phase` without further inversions (calling `null_H`).

    Parameters
    ----------
    H_0_cache : list
        Hamiltonians at different imaginary energies.
    H12 : numpy array
        The hopping matrix between the two cross
        sections of where the SelfEnergyLead is attached.
    phase : float
        Phase at which the supercurrent is calculated.
    params : dict
        A container that is used to store Hamiltonian parameters.

    Returns
    -------
    float
        Total current of all terms in `H_0_list`.
    """
    I = sum(current_contrib_from_H_0(H_0, H12, phase, params)
            for H_0 in H_0_cache)
    return I


def I_c_fixed_n(syst, hopping, params, electron_blocks, matsfreqs=500, N_brute=30):
    H_0_cache = [null_H(syst, params, n, electron_blocks) for n in range(matsfreqs)]
    H12 = hopping(syst, params)
    fun = lambda phase: -current_from_H_0(H_0_cache, H12, phase, params)
    opt = scipy.optimize.brute(
        fun, ranges=[(-np.pi, np.pi)], Ns=N_brute, full_output=True)
    x0, fval, grid, Jout = opt
    return dict(phase_c=x0[0], current_c=-fval, phases=grid, currents=-Jout)


def current_contrib_from_H_0(H_0, H12, phase, params):
    """Uses Dyson’s equation to obtain the Hamiltonian for other
    values of `phase` without further inversions (calling `null_H`).

    Parameters
    ----------
    H_0 : list
        Hamiltonian at a certain imaginary energy.
    H12 : numpy array
        The hopping matrix between the two cross
        sections of where the SelfEnergyLead is attached.
    phase : float
        Phase at which the supercurrent is calculated.
    params : dict
        A container that is used to store Hamiltonian parameters.
    Returns
    -------
    float
        Current contribution of `H_0`.
    """
    t = H12 * np.exp(1j * phase)
    gf = gf_from_H_0(H_0, t - H12)
    dim = t.shape[0]
    H12G21 = t.T.conj() @ gf[dim:, :dim]
    H21G12 = t @ gf[:dim, dim:]
    return -4 * params['T'] * params['current_unit'] * (
        np.trace(H21G12) - np.trace(H12G21)).imag


def current_at_phase(syst, hopping, params, H_0_cache, phase,
                     electron_blocks, tol=1e-2, max_frequencies=100):
    """Find the supercurrent at a phase using a list of Hamiltonians at
    different imaginary energies (Matsubara frequencies). If this list
    does not contain enough Hamiltonians to converge, it automatically
    appends them at higher Matsubara frequencies untill the contribution
    is lower than `tol`, however, it cannot exceed `max_frequencies`.

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem
        The finilized kwant system.
    hopping : function
        Function that returns the hopping matrix between the two cross sections
        of where the SelfEnergyLead is attached.
    params : dict
        A container that is used to store Hamiltonian parameters.
    H_0_cache : list
        Hamiltonians at different imaginary energies.
    phase : float, optional
        Phase at which the supercurrent is calculated.
    electron_blocks: function
        A function that takes the electron blocks from the Hamiltonian.
    tol : float, optional
        Tolerance of the `current_at_phase` function.
    max_frequencies : int, optional
        Maximum number of Matsubara frequencies.

    Returns
    -------
    dict
        Dictionary with the critical phase, critical current, and `currents`
        evaluated at `phases`."""
    if params.get('phase', 0) != 0:
        raise Exception('Set the phase in params to 0!')
    H12 = hopping(syst, params)
    I = 0
    for n in range(max_frequencies):
        if len(H_0_cache) <= n:
            H_0_cache.append(null_H(syst, params, n, electron_blocks))
        I_contrib = current_contrib_from_H_0(H_0_cache[n], H12, phase, params)
        I += I_contrib
        if I_contrib == 0 or tol is not None and abs(I_contrib / I) < tol:
            return I
    # Did not converge within tol using max_frequencies Matsubara frequencies.
    if tol is not None:
        return np.nan
    # if tol is None, return the value after max_frequencies is reached.
    else:
        return I


def I_c(syst, hopping, params, electron_blocks, tol=1e-2,
        max_frequencies=500, N_brute=31):
    """Find the critical current by optimizing the current-phase
    relation.

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem
        The finilized kwant system.
    hopping : function
        Function that returns the hopping matrix between the two cross
        sections of where the SelfEnergyLead is attached.
    params : dict
        A container that is used to store Hamiltonian parameters.
    electron_blocks: function
        A function that takes the electron blocks from the Hamiltonian.
    tol : float, optional
        Tolerance of the `current_at_phase` function.
    max_frequencies : int, optional
        Maximum number of Matsubara frequencies.
    N_brute : int, optional
        Number of points at which the CPR is evaluated in the brute
        force part of the algorithm.

    Returns
    -------
    dict
        Dictionary with the critical phase, critical current, and `currents`
        evaluated at `phases`."""
    H_0_cache = []
    func = lambda phase: -current_at_phase(syst, hopping, params, H_0_cache, phase,
                                           electron_blocks, tol, max_frequencies)
    opt = scipy.optimize.brute(
        func, ranges=((-np.pi, np.pi),), Ns=N_brute, full_output=True)
    x0, fval, grid, Jout = opt
    return dict(phase_c=x0[0], current_c=-fval, phases=grid,
                currents=-Jout, N_freqs=len(H_0_cache))
