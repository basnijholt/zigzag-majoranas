import cmath
from functools import partial, lru_cache
import re

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

    # GET TEMPLATES
    template_barrier, template_normal, template_sc_left, template_sc_right = get_templates(
        a, transverse_soi, mu_from_bottom_of_spin_orbit_bands, k_x_in_sc)

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

    syst.fill(template_normal, shape_normal, (0, a))
    syst.fill(template_barrier, shape_barrier, (0, 0))
    syst.fill(template_barrier, shape_barrier, (0, L_m))

    lead_up = kwant.Builder(kwant.TranslationalSymmetry([0, a]))
    lead_up.fill(template_sc_right, shape_right_sc, (0, 0))

    lead_down = kwant.Builder(kwant.TranslationalSymmetry([0, -a]))
    lead_down.fill(template_sc_left, shape_left_sc, (0, 0))

    # Define left and right cut in the middle of the superconducting part
    cuts = get_sorted_cuts(syst)
    norbs = 4
    if with_vlead:
        syst = supercurrent_matsubara.add_vlead(syst, norbs, *cuts)

    syst.attach_lead(lead_down)
    syst.attach_lead(lead_up)

    syst = syst.finalized()

    hopping = supercurrent_matsubara.hopping_between_cuts(
        syst, *cuts, electron_blocks)
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

    # GET TEMPLATES
    template_barrier, template_normal, template_sc_left, template_sc_right = get_templates(
        a, transverse_soi, mu_from_bottom_of_spin_orbit_bands, k_x_in_sc)

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
        return 0 <= x < L_x and L_m < y < L_m + L_up + 2 * a

    def shape_lead(y1, y2):
        def shape(site):
            (x, y) = site.pos
            return y1 <= y < y2
        return shape

    # BUILD FINITE SYSTEM
    syst = kwant.Builder()

    syst.fill(template_normal, shape_normal, (0, a))
    syst.fill(template_barrier, shape_barrier, (0, 0))
    syst.fill(template_barrier, shape_barrier, (0, L_m))
    if L_down >= a:
        syst.fill(template_sc_left, shape_left_sc, (0, -L_down))
    if L_up >= a:
        syst.fill(template_sc_right, shape_right_sc, (0, L_m + a))

    # LEAD: SLICE OF BULK ALONG X AXIS
    lead = kwant.Builder(kwant.TranslationalSymmetry([-a, 0]))

    lead.fill(template_normal, shape_lead(a, L_m), (0, a))
    lead.fill(template_barrier, shape_lead(0, a), (0, 0))
    lead.fill(template_barrier, shape_lead(L_m, L_m + a), (0, L_m))
    if L_down >= a:
        lead.fill(
            template_sc_left, shape_lead(-L_down - a, 0),
            (0, -L_down - a))
    if L_up >= a:
        lead.fill(template_sc_right, shape_lead(
            L_m + a, L_m + L_up + 2 * a), (0, L_m + a))

    # Define left and right cut in the middle of the superconducting part
    cuts = get_sorted_cuts(syst)
    norbs = 4
    if with_vlead:
        syst = supercurrent_matsubara.add_vlead(syst, norbs, *cuts)

    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    syst = syst.finalized()

    hopping = supercurrent_matsubara.hopping_between_cuts(
        syst, *cuts, electron_blocks)
    return syst, hopping


@lru_cache()
def make_ns_junction(
        a, L_m, L_up, L_down, L_x, transverse_soi=True,
        mu_from_bottom_of_spin_orbit_bands=True, k_x_in_sc=False, **_):
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
    # GET TEMPLATES
    template_barrier, template_normal, template_sc_left, template_sc_right = get_templates(
        a, transverse_soi, mu_from_bottom_of_spin_orbit_bands, k_x_in_sc)

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
    normal_lead.fill(template_normal, shape_lead, (0, 0))

    # Make right superconducting lead
    sc_lead_symmetry = kwant.TranslationalSymmetry((a, 0), (0, a))
    sc_lead = kwant.Builder(sc_lead_symmetry)
    sc_lead.fill(template_sc_right, shape_lead, (0, a))

    # Make barrier/middle site
    wraparound_symmetry = kwant.TranslationalSymmetry((a, 0))
    barrier = kwant.Builder(
        symmetry=wraparound_symmetry,
        conservation_law=conservation_matrix)
    barrier.fill(template_barrier, shape_barrier, (0, 0))

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
    # GET TEMPLATES
    template_barrier, template_normal, template_sc_left, template_sc_right = get_templates(
        a, transverse_soi, mu_from_bottom_of_spin_orbit_bands, k_x_in_sc)

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
        return 0 <= x < L_x and L_m + a <= y < L_m + L_up + 2 * a

    def shape_lead(x1, x2):
        def shape(site):
            (x, y) = site.pos
            return x1 <= y < x2
        return shape

    # BUILD FINITE SYSTEM
    sym = kwant.TranslationalSymmetry((a, 0))
    syst = kwant.Builder(symmetry=sym)

    syst.fill(template_normal, shape_normal, (0, a))
    syst.fill(template_barrier, shape_barrier, (0, 0))
    syst.fill(template_barrier, shape_barrier, (0, L_m))
    if L_down >= a:
        syst.fill(template_sc_left, shape_left_sc, (0, -a))
    if L_up >= a:
        syst.fill(template_sc_right, shape_right_sc, (0, L_m + a))

    syst = kwant.wraparound.wraparound(syst)
    return syst.finalized()


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


@lru_cache()
def make_zigzag_system(
        a, L_m, L_x, L_up, L_down, z_x, z_y, c_x, c_y, edge_thickness=0,
        transverse_soi=True, mu_from_bottom_of_spin_orbit_bands=True,
        k_x_in_sc=True, with_vlead=True, **_):

    # GET TEMPLATES
    template_barrier, template_normal, template_sc_left, template_sc_right = get_templates(
        a, transverse_soi, mu_from_bottom_of_spin_orbit_bands, k_x_in_sc)

    def union_shape(shapes):
        def _shape(pos):
            res = False
            for shape in shapes:
                res |= shape(pos)
            return res
        return _shape

    def intersection_shape(shapes):
        def _shape(pos):
            res = False
            for shape in shapes:
                res &= shape(pos)
            return res
        return _shape

    def difference_shape(shape_A, shape_B):
        def _shape(pos):
            return shape_A(pos) and not shape_B(pos)
        return _shape

    def below_zigzag(z_x, z_y, offset):
        def shape(site):
            x, y = site.pos - offset
            if 0 <= x < 2 * z_x:
                return y < a * ((z_y * np.sin(np.pi * 2 * x / z_x)) // a)
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
            return below_zigzag(z_x, z_y, (offset[0], offset[1] + L_m))(
                pos) and above_zigzag(z_x, z_y, offset)(pos) and 0 <= x < L_x
        return shape

    def edge_zigzag(z_x, z_y, offset):
        def shape(pos):
            x, y = pos.pos
            return ((below_zigzag(z_x, z_y, (offset[0], offset[1] + L_m))(pos) and above_zigzag(z_x, z_y, (offset[0], offset[1] + L_m - a * edge_thickness))(pos) and 0 <= x < L_x)
                    or
                    (below_zigzag(z_x, z_y, (offset[0], offset[1] + edge_thickness * a))(pos) and above_zigzag(z_x, z_y, (offset[0], offset[1]))(pos) and 0 <= x < L_x))
        return shape

    def cut_zigzag(z_x, z_y, offset):
        def shape(pos):
            x, y = pos.pos
            return (below_zigzag(z_x, z_y, (offset[0], offset[1]))(pos) and above_zigzag(
                z_x, z_y, (offset[0], offset[1] - a))(pos) and 0 <= x < L_x)
        return shape

    number_of_zigzags = int(L_x // (2 * z_x)) + 1
    within_zigzag_shapes = [
        within_zigzag(z_x, z_y, (2 * z_x * i, 0))
        for i in range(number_of_zigzags)]
    edge_zigzag_shapes = [
        edge_zigzag(z_x, z_y, (2 * z_x * i, 0))
        for i in range(number_of_zigzags)]
    edge_shape = union_shape(edge_zigzag_shapes)

    cut_zigzag_shapes_up = [
        cut_zigzag(c_x, c_y, (2 * z_x * i, L_m // 2))
        for i in range(number_of_zigzags)]
    cut_shape_up = union_shape(cut_zigzag_shapes_up)
    cut_zigzag_shapes_down = [
        cut_zigzag(
            c_x, c_y, (2 * z_x * i, L_m // 2 - a))
        for i in range(number_of_zigzags)]
    cut_shape_down = union_shape(cut_zigzag_shapes_down)

    middle_shape_with_edge = union_shape(within_zigzag_shapes)
    middle_shape = difference_shape(middle_shape_with_edge, edge_shape)

    def top_shape_block(site):
        x, y = site.pos
        return 0 <= x < L_x and -z_y <= y < L_m + z_y + L_up
    top_shape_with_some_down = difference_shape(
        top_shape_block, middle_shape_with_edge)

    def down_shape_block(site):
        x, y = site.pos
        return 0 <= x < L_x and -L_down - z_y <= y < z_y
    down_shape_with_some_top = difference_shape(
        down_shape_block, middle_shape_with_edge)

    top_shape = top_shape_with_some_down
    down_shape = down_shape_with_some_top

    # BUILD FINITE SYSTEM
    syst = kwant.Builder()
    site_colors = {}

    syst.fill(template_normal, middle_shape, (0, L_m // 2))
    site_colors.update({site: ('black') for site in syst.sites()})

    # CREATE CUT FOR VLEAD
    cut_up = [site for site in syst.sites() if cut_shape_up(site)]
    site_colors.update({site: 'blue' for site in cut_up})
    cut_down = [site for site in syst.sites() if cut_shape_down(site)]
    site_colors.update({site: 'teal' for site in cut_down})
    cuts = [cut_down, cut_up]

    cuts = [sorted(cut, key=lambda s: s.pos[0]) for cut in cuts]
    assert len(cuts[0]) == len(cuts[1]) and len(cuts[0]) > 0, cuts

    norbs = 4
    if with_vlead:
        syst = supercurrent_matsubara.add_vlead(syst, norbs, *cuts)

    # ADD EDGE FOR NON TRANSPARANCY
    if edge_thickness == 0:
        pass

    else:
        for x in np.arange(0, L_x, a):
            y_up = ((z_y * np.sin(np.pi * 2 * x / z_x) + L_m) // a - 1) * a
            y_down = ((z_y * np.sin(np.pi * 2 * x / z_x)) // a) * a

            syst.fill(template_barrier, edge_shape, (x, y_up))
            syst.fill(template_barrier, edge_shape, (x, y_down))

        edge_sites = get_sites_in_shape(syst, edge_shape)
        site_colors.update({site: 'white' for site in edge_sites})

    if L_up is not 0:
        syst.fill(template_sc_left, top_shape, (0, L_m))
    syst.fill(template_sc_right, down_shape, (0, -a))

    sc_sites = get_sites_in_shape(syst, top_shape)
    site_colors.update({site: 'gold' for site in sc_sites})

    sc_sites = get_sites_in_shape(syst, down_shape)
    site_colors.update({site: 'gold' for site in sc_sites})
    syst = syst.finalized()

    hopping = supercurrent_matsubara.hopping_between_cuts(
        syst, *cuts, electron_blocks)
    return syst, site_colors, hopping


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
