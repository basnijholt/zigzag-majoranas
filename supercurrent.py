import numpy as np
import kwant
import sns_system
import spectrum
import numbers
import warnings
from functools import partial

sigz = kwant.continuum.discretizer.ta.array([[1,0,0,0],
                                             [0,1,0,0],
                                             [0,0,-1,0],
                                             [0,0,0,-1]])

def fermi_dirac(e, params):
            if params['T']>0:
                beta = 1/(params['k']*params['T'])
                return np.exp(-beta*e)/(1+np.exp(-beta*e))
            else:
                return np.double(e<=0)
            
def get_cut_sites_and_indices(syst, cut_tag, direction):
    l_cut = []
    r_cut = []
    cut_indices = []
    
    for site_idx, site in enumerate(syst.sites):
        if site.tag[direction]==cut_tag:
            l_cut.append(site)
            temp = [4*site_idx, 4*site_idx+1, 4*site_idx+2, 4*site_idx+3]
            cut_indices.append(temp)
        if site.tag[direction]==cut_tag+1:
            r_cut.append(site)
            temp = [4*site_idx, 4*site_idx+1, 4*site_idx+2, 4*site_idx+3]
            cut_indices.append(temp)
    
    cut_indices = np.hstack(cut_indices)
    cut_sites   = list(zip(l_cut, r_cut))
    
    return (cut_indices, cut_sites)

def ensure_rng(rng=None):
    """Turn rng into a random number generator instance
    If rng is None, return the RandomState instance used by np.random.
    If rng is an integer, return a new RandomState instance seeded with rng.
    If rng is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if rng is None:
        return np.random.mtrand._rand
    if isinstance(rng, numbers.Integral):
        return np.random.RandomState(rng)
    if all(hasattr(rng, attr) for attr in ('random_sample', 'randn',
                                           'randint', 'choice')):
        return rng
    raise ValueError("Expecting a seed or an object that offers the "
                     "numpy.random.RandomState interface.")

def make_projected_current(syst, params, eigvecs, cut=None):
    """Returns a current operator `C` that projects out the Andreev vectors
    on the right side.

    The returned function `f(bra,ket)` gives the output of
    `<bra| C (1-P) |ket>`, where `C` is the current operator,
    and `P` a projector to the Andreev vectors.
    """
    kwant_operator = kwant.operator.Current(syst, sigz, where=cut)
    kwant_operator = kwant_operator.bind(params=params)

    def projected_current(bra, ket):
        nonlocal eigvecs
        projected_ket = eigvecs.T.conj() @ ket
        ket = ket - eigvecs @ projected_ket
        return kwant_operator(bra, ket)
    return projected_current

def make_local_factory(site_indices=None, eigenvecs=None, rng=0, idx=0):
            """Return a `vector_factory` that outputs local vectors.

            If `sites` is provided, the local vectors belong only to
            those sites.
            If `eigenvecs` is provided, project out those vectors form
            the local vectors.
            The parameter `rng` is passed to define a seed for finding the
            bounds of the spectrum. Using the same seed ensures reproducibility.
            """
            rng = ensure_rng(rng)
            if eigenvecs is not None:
                pass
#                 other_vecs = np.zeros((eigenvecs.shape[0], len(site_indices)), dtype=complex)
            
            def vector_factory(n):
                nonlocal idx, rng, eigenvecs, site_indices#, other_vecs
                
                if site_indices is None:
                    site_indices = np.arange(n)
                if idx == -1:
                    idx += 1
                    return np.exp(rng.rand(n) * 2j * np.pi)
                else:
                    vec = np.zeros(n, dtype=complex)
                    vec[site_indices[idx]] = 1

                    if eigenvecs is not None:
                        vec = vec - eigenvecs @ eigenvecs[site_indices[idx],:].conj()
                        vec /= np.sqrt(vec.T.conj() @ vec)
                        if abs(vec[site_indices[idx]])<0.95:
                            warnings.warn('Basis is too non-orthogonal with respect to itself', Warning)
                    idx += 1
                    return vec
            return vector_factory

    
"""
1. Make system
2. Make cut(list of sites, and list of site index)
3. Calculate exact spectrum for k eigenvalues/vectors
4. Calculate current from exact spectrum
    i. Create current operator, bind parameters
    ii. Apply current operator for each eigenstate, apply fermi-dirac
5. Calculate current from kpm part
          i. Create current operator w/ projected out states
         ii. Create eigenvector factory 
        iii. Loop over chunks of the eigenvectors:
             iv. Create SpectralDensity object
              v. Add moments up to correct resolution
             vi. Integrate over spectral density
6. Add both exact and kpm current contributions and sum over cut
"""
def current_kpm_exact(syst_pars, params, k, energy_resolution, cut_tag=0, direction=0, chunk_size=None, operator_ev=True):
    I = 0
    
# 1. Make system
    params.update(dict(**sns_system.constants))
    syst = sns_system.make_sns_system(**syst_pars)
    
# 2. Make cut(list of sites, and list of site index)
    (cut_indices, cut_sites) = get_cut_sites_and_indices(syst, cut_tag, direction)

# 3. Calculate exact spectrum for k eigenvalues/vectors
    ham = syst.hamiltonian_submatrix(params=params, sparse=True)

    (en, evs) = spectrum.sparse_diag(ham, k=k, sigma=0)
    if max(en)<params['Delta']:
        warnings.warn('max(en)<params[\'Delta\']', Warning)
        
# 4. Calculate current from exact spectrum
#  i. Create current operator, bind parameters
    exact_current_operator = kwant.operator.Current(syst, 
                                                    onsite=sigz,
                                                    where=cut_sites
                                                   ).bind(params=params)

    _fermi_dirac = partial(fermi_dirac, params=params)
# ii. Apply current operator for each eigenstate, apply fermi-dirac
    for (e, ev) in zip(en, evs.T):        
        I += _fermi_dirac(e.real) * exact_current_operator(ev)
    
# 5. Calculate current from kpm part
#  i. Create current operator w/ projected out states
    if operator_ev:
        kpm_current_operator = make_projected_current(syst, params, evs, cut=cut_sites)
    else:
        kpm_current_operator = exact_current_operator

# ii. Create eigenvector factory 
    if operator_ev:
        factory = make_local_factory(site_indices=cut_indices)
    else:
        factory = make_local_factory(site_indices=cut_indices, eigenvecs=evs)
    
#iii. Loop over chunks of the vectors:
    if chunk_size==None:
        chunk_size = len(cut_indices)
    
    vectors_left = len(cut_indices)
    while vectors_left>0:
        chunk_size = min(vectors_left, chunk_size)
        
#     iv. Create SpectralDensity object
        sd = kwant.kpm.SpectralDensity(syst,
                                       params=params,
                                       operator=kpm_current_operator,
                                       num_vectors=chunk_size-1,
                                       num_moments=2,
                                       vector_factory=factory)

#      v. Add moments up to correct resolution
        sd.add_moments(energy_resolution=energy_resolution)
        
#     vi. Integrate over spectral density
        
        I += sd.integrate(distribution_function=_fermi_dirac)*chunk_size
        
        vectors_left-=chunk_size
    
    return params['e']/ params['hbar'] * sum(I)

def distributed_current_kpm_exact(syst_pars, params, k, energy_resolution, dview, lview, chunk_size, cut_tag=0, direction=0, operator_projection=True):
    I = 0
    
    params.update(dict(**sns_system.constants))
    syst = sns_system.make_sns_system(**syst_pars)
    
    (cut_indices, cut_sites) = get_cut_sites_and_indices(syst, cut_tag, direction)

    ham = syst.hamiltonian_submatrix(params=params, sparse=True)

    (en, evs) = spectrum.sparse_diag(ham, k=k, sigma=0)
    if max(en)<params['Delta']:
        warnings.warn('max(en)<params[\'Delta\']', Warning)
        
    exact_current_operator = kwant.operator.Current(syst, 
                                                    onsite=sigz,
                                                    where=cut_sites
                                                   ).bind(params=params)

    _fermi_dirac = partial(fermi_dirac, params=params)

    for (e, ev) in zip(en, evs.T):        
        I += _fermi_dirac(e.real) * exact_current_operator(ev)

    chunks = divide_sites_into_chunks(cut_indices, chunk_size)

    filled_in_kpm_calculation = partial(calc_kpm_current,
                                        syst_pars=syst_pars,
                                        params=params,
                                        cut_tag=cut_tag, 
                                        direction=direction,
#                                         evs=evs,
                                        energy_resolution=energy_resolution,
                                        operator_projection=operator_projection)
    
    dview.push(dict(evs= evs))
    
    lview.block = True
    I_kpm = lview.map(filled_in_kpm_calculation, chunks)
#     I_kpm = list(map(filled_in_kpm_calculation, chunks))
    I += np.sum(I_kpm, axis=0)

    return params['e']/ params['hbar'] * sum(I)

  
def divide_sites_into_chunks(vector, chunk_size):
    vector_length = len(vector)
    return [vector[start:start+chunk_size] for start in range(0, vector_length, chunk_size)]

def calc_kpm_current(chunk_indices, syst_pars, params, cut_tag, direction, energy_resolution, operator_projection):
    params.update(dict(**sns_system.constants))
    syst = sns_system.make_sns_system(**syst_pars)

    (cut_indices, cut_sites) = get_cut_sites_and_indices(syst, cut_tag, direction)
    
    _fermi_dirac = partial(fermi_dirac, params=params)

    if(operator_projection):
        kpm_current_operator = make_projected_current(syst, params, evs, cut=cut_sites)
    else:
        kpm_current_operator = kwant.operator.Current(syst, 
                                                    onsite=sigz,
                                                    where=cut_sites
                                                   ).bind(params=params)
    if(operator_projection):
        factory = make_local_factory(site_indices=cut_indices, idx=chunk_indices[0]-cut_indices[0])
    else:
        factory = make_local_factory(site_indices=cut_indices, idx=chunk_indices[0]-cut_indices[0], eigenvecs=evs)

    sd = kwant.kpm.SpectralDensity(syst,
                                       params=params,
                                       operator=kpm_current_operator,
                                       num_vectors=len(chunk_indices)-1,
                                       num_moments=2,
                                       vector_factory=factory)
    sd.add_moments(energy_resolution=energy_resolution)
        
    return sd.integrate(distribution_function=_fermi_dirac)*len(chunk_indices)
        
###################### BAS'S CODE ######################
def get_cuts(syst, lat, first_slice=0, second_slice=None, direction=1):
    """Get the sites at two postions of the specified cut coordinates.

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem
        The finilized kwant system.
    lat : dict
        A container that is used to store Hamiltonian parameters.
    """
    if second_slice is None:
        second_slice = first_slice + 1
    cut_1 = [lat(*tag) for tag in [s.tag for s in syst.sites()]
             if tag[direction] == first_slice]
    cut_2 = [lat(*tag) for tag in [s.tag for s in syst.sites()]
             if tag[direction] == second_slice]
    assert len(cut_1) == len(cut_2), "first_slice and second_slince use site.tag not site.pos!"
    return cut_1, cut_2


def hopping_between_cuts(syst, r_cut, l_cut):
    r_cut_sites = [syst.sites.index(site) for site in r_cut]
    l_cut_sites = [syst.sites.index(site) for site in l_cut]

    def hopping(syst, params):
        return syst.hamiltonian_submatrix(params=params,
                                          to_sites=l_cut_sites,
                                          from_sites=r_cut_sites)[::2, ::2]
    return hopping

def current_at_phase(syst, hopping, params, H_0_cache, phase,
                     tol=1e-2, max_frequencies=500):
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
    tol : float, optional
        Tolerance of the `current_at_phase` function.
    max_frequencies : int, optional
        Maximum number of Matsubara frequencies.

    Returns
    -------
    dict
        Dictionary with the critical phase, critical current, and `currents`
        evaluated at `phases`."""
    try:
        # Make sure that there is no SC phase.
        params['phi'] = 0
    except KeyError:
        pass

    H12 = hopping(syst, params)
    I = 0
    for n in range(max_frequencies):
        if len(H_0_cache) <= n:
            H_0_cache.append(null_H(syst, params, n))
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

def null_H(syst, params, n):
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

    Returns
    -------
    numpy.array
        The Hamiltonian at zero energy and zero phase."""
    en = matsubara_frequency(n, params)
    gf = kwant.greens_function(syst, en, out_leads=[0], in_leads=[0],
                               check_hermiticity=False, params=params)
    return np.linalg.inv(gf.data[::2, ::2])


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

def gf_from_H_0(H_0, t):
    """Returns the Green's function at a phase that is defined inside `t`.
    See doc-string of `current_from_H_0`.
    """
    H = np.copy(H_0)
    dim = t.shape[0]
    H[:dim, dim:] -= t.T.conj()
    H[dim:, :dim] -= t
    return np.linalg.inv(H)





