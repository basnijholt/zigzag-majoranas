import cmath
from functools import partial, lru_cache

import kwant
import numpy as np
import scipy.constants
import topology

import peierls
import supercurrent
import supercurrent_matsubara


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


@lru_cache()
def get_template_strings(
        transverse_soi, mu_from_bottom_of_spin_orbit_bands=True, k_x_in_sc=False, with_k_z=False):
    if mu_from_bottom_of_spin_orbit_bands:
        ham_str = "(hbar^2 / (2*m_eff) * (k_y^2 + k_x^2) - mu + m_eff*alpha_middle^2 / (2 * hbar^2)) * sigma_0 "
    else:
        ham_str = "(hbar^2 / (2*m_eff) * (k_y^2 + k_x^2) - mu) * sigma_0 "

    if with_k_z:
        ham_str += "+ hbar^2 / (2*m_eff) * (k_z^2) * sigma_0"

    if transverse_soi:
        ham_normal = ham_str + """ +
        alpha_middle * (sigma_x * k_y - sigma_y * k_x)"""
        ham_sc_left = ham_str + """
        + alpha_left * (sigma_x * k_y - sigma_y * k_x)"""
        ham_sc_right = ham_str + """
        + alpha_right * (sigma_x * k_y - sigma_y * k_x)"""
    else:
        ham_normal = ham_normal + """
        + alpha_middle * sigma_x * k_y"""
        ham_sc_left = ham_str + """
        + alpha_left * sigma_x * k_y"""
        ham_sc_right = ham_str + """
        + alpha_right * sigma_x * k_y"""

    ham_normal += "+ g_factor_middle*mu_B*B * sigma_x"

    ham_barrier = ham_normal + "+ V * sigma_0"
    ham_sc_left += "+ g_factor_left * mu_B * B * sigma_x"
    ham_sc_right += "+ g_factor_right * mu_B * B * sigma_x"

    if not k_x_in_sc:
        ham_sc_right = ham_sc_right.replace('k_x', '0')
        ham_sc_left = ham_sc_left.replace('k_x', '0')

    template_strings = dict(ham_barrier=ham_barrier,
                            ham_normal=ham_normal,
                            ham_sc_right=ham_sc_right,
                            ham_sc_left=ham_sc_left)
    return template_strings



@lru_cache()
def make_sns_leaded_system(a, L_m, L_x,
                    transverse_soi=True,
                    mu_from_bottom_of_spin_orbit_bands=True,
                    with_vlead=False, k_x_in_sc=False, **_):
    """
    Builds and returns finalized 2dim sns system

    Parameters
    ----------
    a : float
        lattice spacing in nm.
    L_m : float
        width of middle normal strip.
    L_up : float
        width of right superconductor.
    L_down : float
        width of left superconductor.
    L_x : float
        length of finite system.
    Returns
    -------
    syst : kwant.system.FiniteSystem
        Finite system where lead[0] is assumed to be the bulk lead, a slice of the bulk along the y-axis
    """

    #     HAMILTONIAN DEFINITIONS
    template_strings = get_template_strings(
        transverse_soi, mu_from_bottom_of_spin_orbit_bands, k_x_in_sc)

    # TURN HAMILTONIAN STRINGS INTO TEMPLATES
    kwargs = dict(coords=('x', 'y'), grid_spacing=a)
    template_barrier = kwant.continuum.discretize(
        template_strings['ham_barrier'], **kwargs)
    template_normal = kwant.continuum.discretize(
        template_strings['ham_normal'], **kwargs)
    template_sc_left = kwant.continuum.discretize(
        template_strings['ham_sc_left'], **kwargs)
    template_sc_right = kwant.continuum.discretize(
        template_strings['ham_sc_right'], **kwargs)

    # SHAPE FUNCTIONS
    def shape_barrier(site):
        (x, y) = site.pos
        return (0 <= x < L_x) and (y == 0 or y == L_m)

    def shape_normal(site):
        (x, y) = site.pos
        return 0 <= x < L_x and a <= y < L_m

    def shape_left_sc(site):
        (x, y) = site.pos
        return 0 <= x < L_x 

    def shape_right_sc(site):
        (x, y) = site.pos
        return 0 <= x < L_x


    # BUILD FINITE SYSTEM
    syst = kwant.Builder()

    syst.fill(template_normal, shape_normal, (a, 0)[::-1])
    syst.fill(template_barrier, shape_barrier, (0, 0)[::-1])
    syst.fill(template_barrier, shape_barrier, (L_m, 0)[::-1])

    lead_up = kwant.Builder(kwant.TranslationalSymmetry([0, a]))
    lead_up.fill(template_sc_right, shape_right_sc, (0,0))
 
    lead_down = kwant.Builder(kwant.TranslationalSymmetry([0, -a]))
    lead_down.fill(template_sc_left, shape_left_sc, (0,0))

    # Define left and right cut in the middle of the superconducting part
    cuts = supercurrent_matsubara.get_cuts(syst, 0, direction='y')

    # Sort the sites in the `cuts` list.
    cuts = [sorted(cut, key=lambda s: s.pos[0] + s.pos[1]*1e6) for cut in cuts]
    assert len(cuts[0]) == len(cuts[1]) and len(cuts[0]) > 0, cuts
    norbs = 4
    if with_vlead:
        syst = supercurrent_matsubara.add_vlead(syst, norbs, *cuts)

    syst.attach_lead(lead_down)
    syst.attach_lead(lead_up)
    
    syst = syst.finalized()

    electron_blocks = partial(take_electron_blocks, norbs=norbs)
    hopping = supercurrent_matsubara.hopping_between_cuts(syst, *cuts, electron_blocks)

    return syst, hopping



@lru_cache()
def make_sns_system(a, L_m, L_up, L_down, L_x,
                    transverse_soi=True,
                    mu_from_bottom_of_spin_orbit_bands=True,
                    k_x_in_sc=False,
                    with_vlead=False, **_):
    """
    Builds and returns finalized 2dim sns system

    Parameters
    ----------
    a : float
        lattice spacing in nm.
    L_m : float
        width of middle normal strip.
    L_up : float
        width of right superconductor.
    L_down : float
        width of left superconductor.
    L_x : float
        length of finite system.
    Returns
    -------
    syst : kwant.system.FiniteSystem
        Finite system where lead[0] is assumed to be the bulk lead, a slice of the bulk along the y-axis
    """

    #     HAMILTONIAN DEFINITIONS
    template_strings = get_template_strings(
        transverse_soi, mu_from_bottom_of_spin_orbit_bands, k_x_in_sc)

    # TURN HAMILTONIAN STRINGS INTO TEMPLATES
    kwargs = dict(coords=('x', 'y'), grid_spacing=a)
    template_barrier = kwant.continuum.discretize(
        template_strings['ham_barrier'], **kwargs)
    template_normal = kwant.continuum.discretize(
        template_strings['ham_normal'], **kwargs)
    template_sc_left = kwant.continuum.discretize(
        template_strings['ham_sc_left'], **kwargs)
    template_sc_right = kwant.continuum.discretize(
        template_strings['ham_sc_right'], **kwargs)

    # SHAPE FUNCTIONS
    def shape_barrier(site):
        (x, y) = site.pos
        return (0 <= x < L_x) and (y == 0 or y == L_m)

    def shape_normal(site):
        (x, y) = site.pos
        return 0 <= x < L_x and 0 < y < L_m

    def shape_left_sc(site):
        (x, y) = site.pos
        return 0 <= x < L_x and -L_down - a <= y < 0

    def shape_right_sc(site):
        (x, y) = site.pos
        return 0 <= x < L_x and L_m < y < L_m + L_up + 2*a

    def shape_lead(y1, y2):
        def shape(site):
            (x, y) = site.pos
            return y1 <= y < y2
        return shape

    # BUILD FINITE SYSTEM
    syst = kwant.Builder()

    syst.fill(template_normal, shape_normal, (a, 0)[::-1])
    syst.fill(template_barrier, shape_barrier, (0, 0)[::-1])
    syst.fill(template_barrier, shape_barrier, (L_m, 0)[::-1])
    if L_down >= a:
        syst.fill(template_sc_left, shape_left_sc, (-L_down, 0)[::-1])
    if L_up >= a:
        syst.fill(template_sc_right, shape_right_sc, (L_m+a, 0)[::-1])

    # LEAD: SLICE OF BULK ALONG X AXIS
    lead = kwant.Builder(kwant.TranslationalSymmetry([-a, 0]))

    lead.fill(template_normal, shape_lead(a, L_m), (a, 0)[::-1])
    lead.fill(template_barrier, shape_lead(0, a), (0, 0)[::-1])
    lead.fill(template_barrier, shape_lead(L_m, L_m+a), (L_m, 0)[::-1])
    if L_down >= a:
        lead.fill(template_sc_left, shape_lead(-L_down-a, 0), (-L_down-a, 0)[::-1])
    if L_up >= a:
        lead.fill(template_sc_right, shape_lead(L_m+a, L_m + L_up + 2*a), (L_m+a, 0)[::-1])

    # Define left and right cut in the middle of the superconducting part
    cuts = supercurrent_matsubara.get_cuts(syst, 0, direction='y')

    # Sort the sites in the `cuts` list.
    cuts = [sorted(cut, key=lambda s: s.pos[0] + s.pos[1]*1e6) for cut in cuts]
    assert len(cuts[0]) == len(cuts[1]) and len(cuts[0]) > 0, cuts
    norbs = 4
    if with_vlead:
        syst = supercurrent_matsubara.add_vlead(syst, norbs, *cuts)

    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    
    syst = syst.finalized()

    electron_blocks = partial(take_electron_blocks, norbs=norbs)
    hopping = supercurrent_matsubara.hopping_between_cuts(syst, *cuts, electron_blocks)

    return syst, hopping


def take_electron_blocks(H, norbs):
    return H[::2, ::2]


@lru_cache()
def make_ns_junction(a, L_m, L_up, L_down, L_x,
                     transverse_soi=True,
                     mu_from_bottom_of_spin_orbit_bands=True, k_x_in_sc=False,**_):
    """
    Builds and returns finalized NS junction system, for calculating transmission

    Parameters
    ----------
    a : float
        lattice spacing in nm.
    L_m : float
        width of middle normal strip.
    L_up : float
        width of right superconductor.
    L_down : float
        width of left superconductor.
    L_x : float
        length of finite system.
    Returns
    -------
    syst : kwant.system.FiniteSystem
        Finite system where lead[0] is assumed to be the bulk lead,
        a slice of the bulk along the y-axis.
    """

    #     HAMILTONIAN DEFINITIONS
    template_strings = get_template_strings(
        transverse_soi, mu_from_bottom_of_spin_orbit_bands, k_x_in_sc)

    # TURN HAMILTONIAN STRINGS INTO TEMPLATES
    template_barrier = kwant.continuum.discretize(
        template_strings['ham_barrier'], grid_spacing=a)
    template_normal = kwant.continuum.discretize(
        template_strings['ham_normal'], grid_spacing=a)
    template_sc_left = kwant.continuum.discretize(
        template_strings['ham_sc_left'], grid_spacing=a)
    template_sc_right = kwant.continuum.discretize(
        template_strings['ham_sc_right'], grid_spacing=a)

    def shape_barrier(site):
        (x, y) = site.pos
        return x == 0 and 0 <= y < L_m

    def shape_lead(site):
        (x, y) = site.pos
        return x == 0

    # BUILD SYSTEM
    # Allows seperate bookkeeping of eh in normal lead
    conservation_matrix = -supercurrent.sigz

    # Make left normal lead
    normal_lead_symmetry = kwant.TranslationalSymmetry((a, 0), (0, -a))
    normal_lead = kwant.Builder(
        normal_lead_symmetry,
        conservation_law=conservation_matrix)
    normal_lead.fill(template_normal, shape_lead, (0, 0)[::-1])

    # Make right superconducting lead
    sc_lead_symmetry = kwant.TranslationalSymmetry((a, 0), (0, a))
    sc_lead = kwant.Builder(sc_lead_symmetry)
    sc_lead.fill(template_sc_right, shape_lead, (a, 0)[::-1])

    # Make barrier/middle site
    wraparound_symmetry = kwant.TranslationalSymmetry((a, 0))
    barrier = kwant.Builder(
        symmetry=wraparound_symmetry,
        conservation_law=conservation_matrix)
    barrier.fill(template_barrier, shape_barrier, (0, 0)[::-1])

    # Wraparound systems
    barrier = kwant.wraparound.wraparound(barrier)
    normal_lead = kwant.wraparound.wraparound(
        normal_lead, keep=1)  # Keep lead in y-direction
    sc_lead = kwant.wraparound.wraparound(
        sc_lead, keep=1)  # Keep lead in y-direction

    # Attach leads
    barrier.attach_lead(normal_lead)
    barrier.attach_lead(sc_lead)

    return barrier.finalized()


@lru_cache()
def make_wrapped_system(a, L_m, L_up, L_down, L_x,
                        transverse_soi=True,
                        mu_from_bottom_of_spin_orbit_bands=True,
                        k_x_in_sc=False, **_):

    template_strings = get_template_strings(
        transverse_soi, mu_from_bottom_of_spin_orbit_bands, k_x_in_sc)

    # TURN HAMILTONIAN STRINGS INTO TEMPLATES
    template_barrier = kwant.continuum.discretize(
        template_strings['ham_barrier'], grid_spacing=a)
    template_normal = kwant.continuum.discretize(
        template_strings['ham_normal'], grid_spacing=a)
    template_sc_left = kwant.continuum.discretize(
        template_strings['ham_sc_left'], grid_spacing=a)
    template_sc_right = kwant.continuum.discretize(
        template_strings['ham_sc_right'], grid_spacing=a)

    # SHAPE FUNCTIONS
    def shape_barrier(site):
        (x, y) = site.pos
        return (0 <= x < L_x) and (y == 0 or y == L_m)

    def shape_normal(site):
        (x, y) = site.pos
        return 0 <= x < L_x and a <= y < L_m

    def shape_left_sc(site):
        (x, y) = site.pos
        return 0 <= x < L_x and -L_down - a <= y < 0

    def shape_right_sc(site):
        (x, y) = site.pos
        return 0 <= x < L_x and L_m + a <= y < L_m + L_up + 2*a

    def shape_lead(x1, x2):
        def shape(site):
            (x, y) = site.pos
            return x1 <= y < x2
        return shape

    # BUILD FINITE SYSTEM
    sym = kwant.TranslationalSymmetry((a, 0))
    syst = kwant.Builder(symmetry=sym)

    syst.fill(template_normal, shape_normal, (a, 0)[::-1])
    syst.fill(template_barrier, shape_barrier, (0, 0)[::-1])
    syst.fill(template_barrier, shape_barrier, (L_m, 0)[::-1])
    if L_down >= a:
        syst.fill(template_sc_left, shape_left_sc, (-a, 0)[::-1])
    if L_up >= a:
        syst.fill(template_sc_right, shape_right_sc, (L_m+a, 0)[::-1])

    syst = kwant.wraparound.wraparound(syst)
    return syst.finalized()

@lru_cache()
def make_3d_wrapped_system(a, L_m, L_up, L_down, L_x, L_z,
                           transverse_soi=True,
                           mu_from_bottom_of_spin_orbit_bands=True,
                           k_x_in_sc=True, with_vlead=False, **_):

    template_strings = get_template_strings(
        transverse_soi, mu_from_bottom_of_spin_orbit_bands, k_x_in_sc, with_k_z=True)

    # TURN HAMILTONIAN STRINGS INTO TEMPLATES

    def apply_peierls_to_template_string(string, a, dim=3, with_holes=True):
        tb_ham, coords = kwant.continuum.discretize_symbolic(string)
        if dim == 2:
            vector_potential = '[0, 0]'
        elif dim == 3:
            vector_potential = '[0, 0, B * y]'
        tb_ham = peierls.apply(tb_ham, coords,
                               A=vector_potential,
                               signs=[1, -1, 1, -1] if with_holes else None)
        template = kwant.continuum.build_discretized(
            (tb_ham), grid_spacing=a, coords=coords)
        return template

    template_barrier = apply_peierls_to_template_string(template_strings['ham_barrier'], a)
    template_normal = apply_peierls_to_template_string(template_strings['ham_normal'], a)
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
        return (0 <= x < L_x) and (L_m + a <= y < L_m + L_up + 2*a) and (0 <= z < L_z)

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
        syst.fill(template_sc_right, shape_right_sc, (0, L_m+a, 0))

    syst = kwant.wraparound.wraparound(syst)
    # Define left and right cut in the middle of the superconducting part
    cuts = supercurrent_matsubara.get_cuts(syst, 0, direction='y')

    # Sort the sites in the `cuts` list.
    cuts = [sorted(cut, key=lambda s: s.pos[0] + s.pos[1]*1e6) for cut in cuts]
    assert len(cuts[0]) == len(cuts[1]) and len(cuts[0]) > 0, cuts
    norbs = 4
    if with_vlead:
        syst = supercurrent_matsubara.add_vlead(syst, norbs, *cuts)

    syst = syst.finalized()
    electron_blocks = partial(take_electron_blocks, norbs=norbs)
    hopping = supercurrent_matsubara.hopping_between_cuts(syst, *cuts, electron_blocks)

    return syst, hopping

@lru_cache()
def make_zigzag_system(a, L_m, L_x, z_x, z_y, W_up, W_down, edge_thickness=1,
                    transverse_soi=True,
                    mu_from_bottom_of_spin_orbit_bands=True,
                    k_x_in_sc=False, leaded=True, **_):

    #     HAMILTONIAN DEFINITIONS
    template_strings = get_template_strings(
        transverse_soi, mu_from_bottom_of_spin_orbit_bands, k_x_in_sc)
    
    conservation_matrix = -supercurrent.sigz
    
    # TURN HAMILTONIAN STRINGS INTO TEMPLATES
    kwargs = dict(coords=('x', 'y'), grid_spacing=a)
    template_barrier = kwant.continuum.discretize(
        template_strings['ham_barrier'], **kwargs)
    template_normal = kwant.continuum.discretize(
        template_strings['ham_normal'], **kwargs)
    template_sc_left = kwant.continuum.discretize(
        template_strings['ham_sc_left'], **kwargs)
    template_sc_right = kwant.continuum.discretize(
        template_strings['ham_sc_right'], **kwargs)

    def union_shape(shapes):
        def _shape(pos):
            res = False
            for shape in shapes:
                res |= shape(pos)
            return res
        return _shape
        
    def intersection_shape(shape_A, shape_B):
        def _shape(pos):
            return shape_A(pos) and not shape_B(pos)
        return _shape
    
    def below_zigzag(z_x, z_y, offset):
        def shape(site):
            x, y = site.pos - offset
            if 0 <= x < 2*z_x:
                return y <  a * ((z_y* np.sin(np.pi*2 * x / z_x))//a)
            else:
                return False
        return shape
    
    def above_zigzag(z_x, z_y, offset):
        def shape(pos):
            return not below_zigzag(z_x, z_y, offset)(pos)
        return shape
    
    def within_zigzag(z_x, z_y, offset):
        def shape(pos):
            x, y = pos.pos
            return below_zigzag(z_x, z_y, (offset[0], offset[1] + L_m))(pos) and above_zigzag(z_x, z_y, offset)(pos) and 0<=x<L_x
        return shape
    
    def edge_zigzag(z_x, z_y, offset):
        def shape(pos):
            x, y = pos.pos
            return ((below_zigzag(z_x, z_y, (offset[0], offset[1] + L_m))(pos) and above_zigzag(z_x, z_y, (offset[0], offset[1] + L_m-a*edge_thickness))(pos) and 0<=x<L_x)
                    or
                   (below_zigzag(z_x, z_y, (offset[0], offset[1] +edge_thickness*a))(pos) and above_zigzag(z_x, z_y, (offset[0], offset[1]))(pos) and 0<=x<L_x))
        return shape
    
    number_of_zigzags = int(L_x // (2*z_x))+1
    within_zigzag_shapes = [within_zigzag(z_x, z_y, (2*z_x*i, 0)) for i in range(number_of_zigzags)]
    edge_zigzag_shapes = [edge_zigzag(z_x, z_y, (2*z_x*i, 0)) for i in range(number_of_zigzags)]
    edge_shape =union_shape(edge_zigzag_shapes)
    
    middle_shape_with_edge = union_shape(within_zigzag_shapes)
    middle_shape = intersection_shape(middle_shape_with_edge, edge_shape)
    
    def top_shape_block(site):
        x, y = site.pos
        return 0 <= x < L_x and -z_y <= y < L_m + z_y + W_up
    top_shape_with_some_down = intersection_shape(top_shape_block, middle_shape_with_edge)
    
    def down_shape_block(site):
        x, y = site.pos
        return 0 <= x < L_x and -W_down - z_y <= y < z_y
    down_shape_with_some_top = intersection_shape(down_shape_block, middle_shape_with_edge)
    
    top_shape = top_shape_with_some_down
#     intersection_shape(top_shape_with_some_down, down_shape_with_some_top)
    down_shape = down_shape_with_some_top
#     intersection_shape(down_shape_with_some_top, top_shape)
    
    # BUILD FINITE SYSTEM
    syst = kwant.Builder()
    syst.fill(template_normal, middle_shape, (0, L_m//2))
    
    if edge_thickness ==0:
        pass
    
    else:
        for x in np.arange(0, L_x, a):
            y_up = ((z_y* np.sin(np.pi*2 * x / z_x)+L_m)//a - 1)*a
            y_down = ((z_y* np.sin(np.pi*2 * x / z_x))//a)*a
#             if (x//z_x)%2 == 1:
#                 y = z_y - y
            syst.fill(template_barrier, edge_shape, (x, y_up))
            syst.fill(template_barrier, edge_shape, (x, y_down))
       
    if W_up is not 0:
        syst.fill(template_sc_left, top_shape, (0, L_m))
    syst.fill(template_sc_right, down_shape, (0, -a))
    
    syst = syst.finalized()

    return syst

def to_site_ph_spin(syst_pars, wf):
    norbs = 4
    nsites = len(wf) // norbs
    nsitesL = int(syst_pars['L_x'] / syst_pars['a'])
    nsitesW = nsites // nsitesL

    wf_eh_sp = np.reshape(wf, (nsites, norbs))
    wf_eh_sp_grid = np.reshape(wf_eh_sp, (nsitesW, nsitesL, norbs))

    return wf_eh_sp_grid