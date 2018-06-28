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
                res= np.exp(-beta*e)/(1+np.exp(-beta*e))
                if np.any(np.isnan(res)):
                    res = np.double(e<=0)
                return res
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

def make_local_factory(site_indices=None, eigenvecs=None, rng=0):
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

            idx = -1
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

    
#iii. Loop over chunks of the vectors:
    if chunk_size is None:
        chunk_size = len(cut_indices)
    
    vectors_left = len(cut_indices)
    chunks_inidices = divide_sites_into_chunks(cut_indices, chunk_size)
    for chunk in chunks_inidices:
        chunk_size = len(chunk)
        
    # ii. Create eigenvector factory 
        if operator_ev:
            factory = make_local_factory(site_indices=chunk)
        else:
            factory = make_local_factory(site_indices=chunk, eigenvecs=evs)
#     iv. Create SpectralDensity object
        sd = kwant.kpm.SpectralDensity(syst,
                                       params=params,
                                       operator=kpm_current_operator,
                                       num_vectors=chunk_size,
                                       num_moments=2,
                                       vector_factory=factory)

#      v. Add moments up to correct resolution
        sd.add_moments(energy_resolution=energy_resolution)
        
#     vi. Integrate over spectral density
        
        I += sd.integrate(distribution_function=_fermi_dirac)*chunk_size
    
    return params['e']/ params['hbar'] * sum(I)

def distributed_current_kpm_exact(syst_pars, params, k, energy_resolution, dview, lview, chunk_size, cut_tag=0, direction=0, operator_projection=True):
    from time import time
    I = 0
    
    lview.block = True
    dview.block = True
    
    params.update(dict(**sns_system.constants))
    syst = sns_system.make_sns_system(**syst_pars)
    
    (cut_indices, cut_sites) = get_cut_sites_and_indices(syst, cut_tag, direction)

    ham = syst.hamiltonian_submatrix(params=params, sparse=True)

    t=time()
    dview.apply(calc_spectrum, syst_pars, params, k)
    t2=time()
    print('Spectrum calculation on nodes',t2-t)

    t=time()
    (en, evs) = spectrum.sparse_diag(ham, k=k, sigma=0)
    t2=time()
    print('Spectrum calculation on io', t2-t)

    
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

    filled_in_current = partial(calc_kpm_current,
                              syst_pars=syst_pars,
                              params=params,
                              cut_tag=cut_tag,
                              direction=direction,
                              energy_resolution=energy_resolution,
                              operator_projection=operator_projection)
    
#     I_kpm = list(map(filled_in_kpm_calculation, chunks))
    t=time()
    I_kpm = lview.map(filled_in_current, chunks)
    t2=time()
    print('KPM calculation', t2-t)


    I += np.sum(I_kpm, axis=0)

    return params['e']/ params['hbar'] * sum(I)


def divide_sites_into_chunks(vector, chunk_size):
    vector_length = len(vector)
    return [vector[start:start+chunk_size] for start in range(0, vector_length, chunk_size)]

def calc_kpm_current(chunk_indices, syst_pars, params, cut_tag, direction, energy_resolution):
    
    
    params.update(dict(**sns_system.constants))
    syst = sns_system.make_sns_system(**syst_pars)

    (cut_indices, cut_sites) = get_cut_sites_and_indices(syst, cut_tag, direction)
    
    _fermi_dirac = partial(fermi_dirac, params=params)

    kpm_current_operator = kwant.operator.Current(syst, 
                                                    onsite=sigz,
                                                    where=cut_sites
                                                   ).bind(params=params)
    
    factory = make_local_factory(site_indices=chunk_indices)


    sd = kwant.kpm.SpectralDensity(syst,
                                       params=params,
                                       operator=kpm_current_operator,
                                       num_vectors=len(chunk_indices),
                                       num_moments=2,
                                       vector_factory=factory)
    sd.add_moments(energy_resolution=energy_resolution)
        
    return sd.integrate(distribution_function=_fermi_dirac)*len(chunk_indices)

def make_ev_factory(eigenvecs, rng=0):
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

            idx = -1
            def vector_factory(n):
                nonlocal idx, rng, eigenvecs
                
                if idx == -1:
                    idx += 1
                    return np.exp(rng.rand(n) * 2j * np.pi)
                else:
                    vec = eigenvecs[:,idx]
                    idx += 1
                    return vec
            return vector_factory

def calc_kpm_current_evs(evs, syst_pars, params, cut_tag, direction, energy_resolution): 
    evs = evs.copy()
    factory = make_ev_factory(evs)
    
    params.update(dict(**sns_system.constants))
    syst = sns_system.make_sns_system(**syst_pars)

    (cut_indices, cut_sites) = get_cut_sites_and_indices(syst, cut_tag, direction)
    
    _fermi_dirac = partial(fermi_dirac, params=params)


    exact_current_operator = kwant.operator.Current(syst, 
                                                onsite=sigz,
                                                where=cut_sites
                                               ).bind(params=params)

    sd = kwant.kpm.SpectralDensity(syst,
                                   params=params,
                                   operator=exact_current_operator,
                                   num_vectors=evs.shape[1],
                                   num_moments=2,
                                   vector_factory=factory)

#      v. Add moments up to correct resolution
    sd.add_moments(energy_resolution=energy_resolution)

    return sd.integrate(distribution_function=_fermi_dirac)*evs.shape[1]
        
def current_kpm_non_projected(syst_pars, params,
                              k, energy_resolution, lview, chunk_size,
                              cut_tag=0, direction=0):
    lview.block = False

    params.update(dict(**sns_system.constants))
    syst = sns_system.make_sns_system(**syst_pars)
    
    _fermi_dirac = partial(fermi_dirac, params=params)

    (cut_indices, cut_sites) = get_cut_sites_and_indices(syst, cut_tag, direction)

    chunks = divide_sites_into_chunks(cut_indices, chunk_size)

    filled_in_calc_kpm_current = partial(calc_kpm_current,
                                         syst_pars=syst_pars,
                                         params=params,
                                         cut_tag=cut_tag,
                                         direction=direction,
                                         energy_resolution=energy_resolution)
    
    ar_current_all_kpm = lview.map(filled_in_calc_kpm_current, chunks)


    ham = syst.hamiltonian_submatrix(params=params, sparse=True)
    (en, evs) = spectrum.sparse_diag(ham, k=k, sigma=0)

    if max(en)<params['Delta']:
        delta = params['Delta']
        warnings.warn(f'max(en)<params[\'Delta\'] ({max(en)}<{delta})', Warning)
        

    filled_in_calc_kpm_current_evs = partial(calc_kpm_current_evs,
                                             syst_pars=syst_pars,
                                             params=params,
                                             cut_tag=cut_tag,
                                             direction=direction,
                                             energy_resolution=energy_resolution) 

    ev_chunk_indices = divide_sites_into_chunks(np.arange(evs.shape[1]), chunk_size)
    ev_chunks = tuple(evs[:,chunk_ind].copy() for chunk_ind in ev_chunk_indices)
 
    ar_ABS_kpm = lview.map(filled_in_calc_kpm_current_evs, ev_chunks)

#############################################################
#  Calc exact current contribution
#############################################################
    exact_current_operator = kwant.operator.Current(syst, 
                                                onsite=sigz,
                                                where=cut_sites
                                               ).bind(params=params)

    I_AB_exact = 0
    for (e, ev) in zip(en, evs.T):        
        I_AB_exact += _fermi_dirac(e.real) * exact_current_operator(ev)

#######################################################
# Wait until all done
#######################################################

    I_all_kpm  = sum(ar_current_all_kpm.get())
    I_AB_kpm = sum(ar_ABS_kpm.get())

    return params['e']/ params['hbar'] * ((I_AB_exact) + (I_all_kpm - (I_AB_kpm)))

def get_cuts(*a, **_):
    pass