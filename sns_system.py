import cmath
from math import cos, sin
import math
from functools import partial, lru_cache
import re

import kwant
import numpy as np
import scipy.constants
import scipy.interpolate
from scipy.optimize import fsolve

import peierls
import supercurrent
import supercurrent_matsubara
import topology
from shape import *


constants = dict(
    # effective mass in kg,
    m_eff=0.023 * scipy.constants.m_e / (scipy.constants.eV * 1e-3) / 1e18,
    hbar=scipy.constants.hbar / (scipy.constants.eV * 1e-3),
    mu_B=scipy.constants.physical_constants['Bohr magneton'][0] / (
        scipy.constants.eV * 1e-3),
    exp=cmath.exp,
    cos=cmath.cos,
    sin=cmath.sin)

dummy_params_raw = dict(g_factor_middle=1,
                        g_factor_left=2,
                        g_factor_right=3,
                        mu=4,
                        alpha_middle=5,
                        alpha_left=6,
                        alpha_right=7,
                        Delta_left=8,
                        Delta_right=9,
                        B=10,
                        phase=11,
                        T=12,
                        V=13)

dummy_params = dict(**constants,
                    **dummy_params_raw)

def create_parallel_sine(distance, z_x, z_y):
    def _parallel_sine(x, distance, z_x, z_y):
        g       = lambda t: z_y * math.sin(2*np.pi/z_x*t)
        g_prime = lambda t: z_y * 2*np.pi/z_x*math.cos(2*np.pi/z_x*t)
        def _x(t):
            return t - distance*g_prime(t)/np.sqrt(1 + g_prime(t)**2) - x

        def y(t):
            return g(t) + distance/np.sqrt(1 + g_prime(t)**2)

        t = fsolve(_x, x)
        return y(t)

    xdim = np.linspace(0, z_x, 1000)
    ydim = [_parallel_sine(x, distance, z_x, z_y) for x in xdim]
    
    parallel_sine = scipy.interpolate.interp1d(xdim, ydim)
    
    return lambda x: parallel_sine(x%z_x)

def remove_phs(H):
    return re.sub(r'kron\((sigma_[xyz0]), sigma_[xzy0]\)', r'\1', H)


@lru_cache()
def get_template_strings(
        transverse_soi, mu_from_bottom_of_spin_orbit_bands=True,
        k_x_in_sc=False, with_k_z=False, no_phs=False):
    kinetic = "(hbar^2 / (2*m_eff) * (k_y^2 + k_x^2 + k_z^2) - mu {}) * kron(sigma_0, sigma_z)"
    if mu_from_bottom_of_spin_orbit_bands:
        ham_str = kinetic.format("+ m_eff*alpha_middle^2 / (2 * hbar^2)")
    else:
        ham_str = kinetic.format("")

    if not with_k_z:
        ham_str = ham_str.replace('k_z', '0')

    spin_orbit = """+ alpha_{} * kron(sigma_x, sigma_z) * k_y"""
    ham_normal = ham_str + spin_orbit.format('middle')
    ham_sc_left = ham_str + spin_orbit.format('left')
    ham_sc_right = ham_str + spin_orbit.format('right')

    if transverse_soi:
        tr_spin_orbit = "- alpha_{} * kron(sigma_y, sigma_z) * k_x"
        ham_normal += tr_spin_orbit.format('middle')
        ham_sc_left += tr_spin_orbit.format('left')
        ham_sc_right += tr_spin_orbit.format('right')

    superconductivity = """+ Delta_{0} * (cos({1}phase / 2) * kron(sigma_0, sigma_x)
                                          + sin({1}phase / 2) * kron(sigma_0, sigma_y))"""
    ham_sc_left += superconductivity.format('left', '-')
    ham_sc_right += superconductivity.format('right', '+')

    zeeman = "+ g_factor_{} * mu_B * B * kron(sigma_x, sigma_0)"
    ham_normal += zeeman.format('middle')
    ham_sc_left += zeeman.format('left')
    ham_sc_right += zeeman.format('right')

    ham_barrier = ham_normal + "+ V * kron(sigma_0, sigma_z)"

    if not k_x_in_sc:
        ham_sc_right = ham_sc_right.replace('k_x', '0')
        ham_sc_left = ham_sc_left.replace('k_x', '0')

    template_strings = dict(ham_barrier=ham_barrier,
                            ham_normal=ham_normal,
                            ham_sc_right=ham_sc_right,
                            ham_sc_left=ham_sc_left)

    if no_phs:
        template_strings = {k: remove_phs(v) for k, v in template_strings.items()}

    return template_strings


@lru_cache()
def get_templates(a, transverse_soi, mu_from_bottom_of_spin_orbit_bands, k_x_in_sc):
    template_strings = get_template_strings(
        transverse_soi, mu_from_bottom_of_spin_orbit_bands, k_x_in_sc)
    kwargs = dict(coords=('x', 'y'), grid_spacing=a)
    template_barrier = kwant.continuum.discretize(
        template_strings['ham_barrier'], **kwargs)
    template_normal = kwant.continuum.discretize(
        template_strings['ham_normal'], **kwargs)
    template_sc_left = kwant.continuum.discretize(
        template_strings['ham_sc_left'], **kwargs)
    template_sc_right = kwant.continuum.discretize(
        template_strings['ham_sc_right'], **kwargs)
    return (template_barrier, template_normal,
            template_sc_left, template_sc_right)


def get_sorted_cuts(syst):
    cuts = supercurrent_matsubara.get_cuts(syst, 0, direction='y')

    # Sort the sites in the `cuts` list.
    cuts = [sorted(cut, key=lambda s: s.pos[0] + s.pos[1] * 1e6)
            for cut in cuts]
    assert len(cuts[0]) == len(cuts[1]) and len(cuts[0]) > 0, cuts
    return cuts


def electron_blocks(H):
    return H[::2, ::2]


@lru_cache()
def make_3d_wrapped_system(a, L_m, L_up, L_down, L_x, L_z, with_orbital,
                           transverse_soi=True,
                           mu_from_bottom_of_spin_orbit_bands=True,
                           k_x_in_sc=True, with_vlead=False, **_):

    template_strings = get_template_strings(
        transverse_soi,
        mu_from_bottom_of_spin_orbit_bands,
        k_x_in_sc,
        with_k_z=True)

    # TURN HAMILTONIAN STRINGS INTO TEMPLATES

    def apply_peierls_to_template_string(string, a, dim=3, with_holes=True):
        tb_ham, coords = kwant.continuum.discretize_symbolic(string)
        if dim == 2:
            vector_potential = '[0, 0]'
        elif dim == 3:
            vector_potential = '[0, 0, B * y]'
        if with_orbital:
            signs = [1, -1, 1, -1] if with_holes else None
            tb_ham = peierls.apply(
                tb_ham, coords, A=vector_potential, signs=signs)
        template = kwant.continuum.build_discretized(
            (tb_ham), grid_spacing=a, coords=coords)
        return template

    template_barrier = apply_peierls_to_template_string(
        template_strings['ham_barrier'], a)
    template_normal = apply_peierls_to_template_string(
        template_strings['ham_normal'], a)
    template_sc_left = kwant.continuum.discretize(
        template_strings['ham_sc_left'], grid_spacing=a)
    template_sc_right = kwant.continuum.discretize(
        template_strings['ham_sc_right'], grid_spacing=a)

    # SHAPE FUNCTIONS
    def shape_barrier(site):
        (x, y, z) = site.pos
        return (0 <= x < L_x) and (y == 0 or y == L_m) and (0 <= z < L_z)

    def shape_normal(site):
        (x, y, z) = site.pos
        return (0 <= x < L_x) and (a <= y < L_m) and (0 <= z < L_z)

    def shape_left_sc(site):
        (x, y, z) = site.pos
        return (0 <= x < L_x) and (-L_down - a <= y < 0) and (0 <= z < L_z)

    def shape_right_sc(site):
        (x, y, z) = site.pos
        return (0 <= x < L_x) and (L_m + a <= y <
                                   L_m + L_up + 2 * a) and (0 <= z < L_z)

    def shape_lead(x1, x2):
        def shape(site):
            (x, y, z) = site.pos
            return (x1 <= y < x2) and (0 <= z < L_z)
        return shape

    # BUILD FINITE SYSTEM
    sym = kwant.TranslationalSymmetry((a, 0, 0))
    syst = kwant.Builder(symmetry=sym)

    syst.fill(template_normal, shape_normal, (0, a, 0))
    syst.fill(template_barrier, shape_barrier, (0, 0, 0))
    syst.fill(template_barrier, shape_barrier, (0, L_m, 0))
    if L_down >= a:
        syst.fill(template_sc_left, shape_left_sc, (0, -a, 0))
    if L_up >= a:
        syst.fill(template_sc_right, shape_right_sc, (0, L_m + a, 0))

    syst = kwant.wraparound.wraparound(syst)
    # Define left and right cut in the middle of the superconducting part
    cuts = get_sorted_cuts(syst)
    norbs = 4
    if with_vlead:
        syst = supercurrent_matsubara.add_vlead(syst, norbs, *cuts)

    syst = syst.finalized()
    hopping = supercurrent_matsubara.hopping_between_cuts(
        syst, *cuts, electron_blocks)
    return syst, hopping



def get_sites_in_shape(syst, shape):
    sites = []
    for site in syst.sites():
        if shape(site):
            sites.append(site)
    return sites


def to_site_ph_spin(syst_pars, wf):
    norbs = 4
    nsites = len(wf) // norbs
    nsitesL = int(syst_pars['L_x'] / syst_pars['a'])
    nsitesW = nsites // nsitesL

    wf_eh_sp = np.reshape(wf, (nsites, norbs))
    wf_eh_sp_grid = np.reshape(wf_eh_sp, (nsitesW, nsitesL, norbs))

    return wf_eh_sp_grid

@lru_cache()
def make_system(L_m, L_x, L_sc_up, L_sc_down, z_x, z_y, a,
                sawtooth,
                parallel_curve,
                transverse_soi,
                mu_from_bottom_of_spin_orbit_bands,
                k_x_in_sc,
                wraparound,
                current,
                ns_junction):
    
    ######################
    ## Define templates ##
    ######################
    assert((sawtooth and parallel_curve) is not True)
    
    template_barrier, template_normal, template_sc_left, template_sc_right = get_templates(
        a, transverse_soi, mu_from_bottom_of_spin_orbit_bands, k_x_in_sc)

    template_interior = template_normal
    template_edge = template_barrier
    template_top_superconductor = template_sc_left
    template_bottom_superconductor = template_sc_right

    
    if parallel_curve:
        curve = create_parallel_sine(0, z_x, z_y)

        curve_top = create_parallel_sine(L_m//2, z_x, z_y)
        below_shape = below_curve(curve_top)

        curve_bottom = create_parallel_sine(-L_m//2, z_x, z_y)
        above_shape = above_curve(curve_bottom)
        
    elif sawtooth:
        _curve = lambda x: 4*z_y/z_x*(x%(z_x//2)) - z_y if x%z_x < z_x//2 else -4*z_y/z_x*(x%(z_x//2)) + z_y
        curve = lambda x: _curve(x+z_x/4)
        
        if z_y is not 0:
            theta = np.arctan(4*z_y/z_x)
            y_offset = L_m/np.cos(theta)
        else:
            y_offset = 0
            
        below_shape = below_curve(lambda x: curve(x) + y_offset//2)
        above_shape = above_curve(lambda x: curve(x) - y_offset//2)        
        
    else:
        curve = lambda x: z_y*sin(2*np.pi / z_x * x)
        below_shape = below_curve(lambda x: curve(x) + L_m//2)
        above_shape = above_curve(lambda x: curve(x) - L_m//2)


    #--------------
    # Define middle
    middle_shape = (below_shape * above_shape)[0:L_x, :]
    
    #------------
    # Define edge
    edge_shape = middle_shape.edge()
    edge_initial_site = (0, 0)

    #------------------------
    # Remove edge from middle
    interior_shape = middle_shape.interior()
    interior_initial_site = (a, a)

    #--------------------------
    # Define top superconductor
    if sawtooth:
        top_superconductor_initial_site = (0, y_offset//2+a)
        top_superconductor_shape = middle_shape.inverse()[0:L_x, :L_sc_up + y_offset//2 + z_y]
    else:
        top_superconductor_initial_site = (z_x//4, L_m//2+z_y+a)
        top_superconductor_shape = middle_shape.inverse()[0:L_x, :L_sc_up + L_m//2 + z_y]
    
    #-----------------------------
    # Define bottom superconductor
    if sawtooth:
        bottom_superconductor_initial_site = (0, -y_offset//2-a)
        bottom_superconductor_shape = middle_shape.inverse()[0:L_x, -L_sc_down - y_offset//2 - z_y:]
    else:
        bottom_superconductor_initial_site = (z_x//4, -L_m//2-z_y-2*a)
        bottom_superconductor_shape = middle_shape.inverse()[0:L_x, -L_sc_down - L_m//2 - z_y:]

    #-----------
    # Define cut
    cut_curve_top = above_curve(lambda x: curve(x))
    cut_curve_bottom = below_curve(lambda x: curve(x))

    top_cut_shape = above_curve(curve).edge()
    bottom_cut_shape = below_curve(curve).edge()

    #-------------------
    # NS junction shapes
    if ns_junction:
        middle_shape = below_curve(curve).edge()[0:L_x, :]
        top_superconductor_lead_shape = Shape()[0:L_x, :]
        bottom_normal_lead_shape = Shape()[0:L_x, :]
        
    ############################
    ## Build junnction system ##
    ############################
    site_colors = dict()
    if ns_junction:
        conservation_matrix = -supercurrent.sigz
        
        barrier_syst = kwant.Builder(conservation_law=conservation_matrix)
        top_superconductor_lead = kwant.Builder(kwant.TranslationalSymmetry((0, a)))
        bottom_normal_lead = kwant.Builder(kwant.TranslationalSymmetry((0, -a)),
                                           conservation_law=conservation_matrix)
        
        edge_sites = barrier_syst.fill(template_edge, middle_shape, (0, -a))
        add_to_site_colors(site_colors, edge_sites, 'middle_barrier')
        
        
        top_superconductor_lead.fill(template_top_superconductor, 
                                       top_superconductor_lead_shape, 
                                       (0,0))
        
        bottom_normal_lead.fill(template_normal,
                                 bottom_normal_lead_shape, 
                                 (0,0))
        
        top_superconductor_sites = barrier_syst.attach_lead(top_superconductor_lead)
        add_to_site_colors(site_colors, top_superconductor_sites, 'top_superconductor')

        
        normal_sites = barrier_syst.attach_lead(bottom_normal_lead)
        add_to_site_colors(site_colors, normal_sites, 'middle_interior')

      
        barrier_syst = barrier_syst.finalized()
        return barrier_syst, site_colors, None
    
    ##################
    ## Build system ##
    ##################        
    else:
        #---------------
        # Create builder
        if wraparound:
            syst = kwant.Builder(kwant.TranslationalSymmetry([L_x, 0]))
        else:
            syst = kwant.Builder()

        #----------
        # Fill edge
        edge_sites = syst.fill(template_edge, edge_shape, edge_initial_site)
        add_to_site_colors(site_colors, edge_sites, 'middle_barrier')

        #--------------
        # Fill interior
        interior_sites = syst.fill(template_interior, interior_shape, interior_initial_site)
        add_to_site_colors(site_colors, interior_sites, 'middle_interior')

        #------------------------
        # Fill top superconductor
        if L_sc_up is not 0:
            top_superconductor_sites = syst.fill(template_top_superconductor, top_superconductor_shape, top_superconductor_initial_site)
            add_to_site_colors(site_colors, top_superconductor_sites, 'top_superconductor')

        #---------------------------
        # Fill bottom superconductor
        if L_sc_down is not 0:
            bottom_superconductor_sites = syst.fill(template_bottom_superconductor, bottom_superconductor_shape, bottom_superconductor_initial_site)
            add_to_site_colors(site_colors, bottom_superconductor_sites, 'bottom_superconductor')


        ##########################################
        ## Add features for current calculation ##
        ##########################################

        #---------------------
        # Make cut for current
        if current:
            top_cut = list(site for site in syst.sites() if top_cut_shape(site))
            add_to_site_colors(site_colors, top_cut, 'top_cut')
            bottom_cut = list(site for site in syst.sites() if bottom_cut_shape(site))
            add_to_site_colors(site_colors, bottom_cut, 'bottom_cut')

            cuts = (top_cut, bottom_cut)

            norbs = 4
            syst = supercurrent_matsubara.add_vlead(syst, norbs, *cuts)

        #####################
        ## Finalize system ##
        #####################

        if wraparound:
            syst = kwant.wraparound.wraparound(syst)

        syst = syst.finalized()

        #---------------------
        # Get hopping
        if current:
            hopping = supercurrent_matsubara.hopping_between_cuts(syst, *cuts, electron_blocks)
            return syst, site_colors, hopping
        else:
            return syst, site_colors, None