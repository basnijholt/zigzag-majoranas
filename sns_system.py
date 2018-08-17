import kwant
import topology
import cmath
import scipy.constants
import numpy as np
import supercurrent
from functools import lru_cache

constants = dict(
    m_eff=0.023 * scipy.constants.m_e / (scipy.constants.eV * 1e-3) / 1e18,  # effective mass in kg, 
    hbar=scipy.constants.hbar / (scipy.constants.eV * 1e-3),
    mu_B=scipy.constants.physical_constants['Bohr magneton'][0] / (scipy.constants.eV * 1e-3),
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
def get_template_strings(transverse_soi, mu_from_bottom_of_spin_orbit_bands=True):
    if mu_from_bottom_of_spin_orbit_bands:
        ham_str = "(hbar^2 / (2*m_eff) * (k_x^2 + k_y^2) - mu + m_eff*alpha_middle^2 / (2 * hbar^2)) * kron(sigma_z, sigma_0) "
    else:
        ham_str = "(hbar^2 / (2*m_eff) * (k_x^2 + k_y^2) - mu) * kron(sigma_z, sigma_0) "

    if transverse_soi:
        ham_normal = ham_str + """ +
        alpha_middle * (kron(sigma_z, sigma_x) * k_y - kron(sigma_z, sigma_y) * k_x)"""
        ham_sc_left = ham_str + """
        + alpha_left * (kron(sigma_z, sigma_x) * k_y - kron(sigma_z, sigma_y) * k_x)"""
        ham_sc_right = ham_str + """
        + alpha_right * (kron(sigma_z, sigma_x) * k_y - kron(sigma_z, sigma_y) * k_x)"""
    else:
        ham_normal = ham_normal + """
        + alpha_middle * kron(sigma_z, sigma_x) * k_y"""
        ham_sc_left = ham_str + """
        + alpha_left * kron(sigma_z, sigma_x) * k_y"""
        ham_sc_right = ham_str + """
        + alpha_right * kron(sigma_z, sigma_x) * k_y"""

    ham_sc_left += " + Delta_left * kron(sigma_y, sigma_0)"
    ham_sc_right += """ + cos(phase) * Delta_right * kron(sigma_y, sigma_0) + 
                                    + sin(phase) * Delta_right * kron(sigma_x, sigma_0)
    """
    ham_normal += "+ g_factor_middle*mu_B*B * kron(sigma_0, sigma_y)"

    ham_barrier = ham_normal + "+ V * kron(sigma_z, sigma_0)"
    ham_sc_left  += "+ g_factor_left * mu_B * B * kron(sigma_0, sigma_y)"
    ham_sc_right += "+ g_factor_right * mu_B * B * kron(sigma_0, sigma_y)"
    
    template_strings = dict(ham_barrier=ham_barrier,
                            ham_normal=ham_normal,
                            ham_sc_right=ham_sc_right,
                            ham_sc_left=ham_sc_left)
    return template_strings

@lru_cache()
def make_sns_system(a, Lm, Lr, Ll, Ly,
                    transverse_soi = True,
                    mu_from_bottom_of_spin_orbit_bands = True):
    """ 
    Builds and returns finalized 2dim sns system
    
    Parameters:
    -----------
    a : float
        lattice spacing in nm

    Lm : float
        width of middle normal strip

    Lr : float
        width of right superconductor

    Ll : float
        width of left superconductor

    Ly : float
        length of finite system

    Returns:
    --------
    syst : kwant.system.FiniteSystem
        Finite system where lead[0] is assumed to be the bulk lead, a slice of the bulk along the y-axis
    """

    #     HAMILTONIAN DEFINITIONS
    template_strings = get_template_strings(transverse_soi, mu_from_bottom_of_spin_orbit_bands)

    # TURN HAMILTONIAN STRINGS INTO TEMPLATES
    template_barrier = kwant.continuum.discretize(template_strings['ham_barrier'], grid_spacing=a)
    template_normal = kwant.continuum.discretize(template_strings['ham_normal'], grid_spacing=a)
    template_sc_left = kwant.continuum.discretize(template_strings['ham_sc_left'], grid_spacing=a)
    template_sc_right = kwant.continuum.discretize(template_strings['ham_sc_right'], grid_spacing=a)
    
    # SHAPE FUNCTIONS
    def shape_barrier(site):
        (x, y) = site.pos
        return (0 <= y < Ly) and (x == 0 or x== Lm - a)

    def shape_normal(site):
        (x, y) = site.pos
        return 0 <= y < Ly and a <= x < Lm - a
    
    def shape_left_sc(site):
        (x, y) = site.pos
        return 0 <= y < Ly and -Ll <= x < 0
    
    def shape_right_sc(site):
        (x, y) = site.pos
        return 0 <= y < Ly and Lm <= x < Lm + Lr

    def shape_lead(x1, x2):
        def shape(site):
            (x, y) = site.pos
            return  x1 <= x < x2
        return shape
    
    # BUILD FINITE SYSTEM
    syst = kwant.Builder()

    syst.fill(template_normal, shape_normal, (a,0))
    syst.fill(template_barrier, shape_barrier, (0,0))
    syst.fill(template_barrier, shape_barrier, (Lm-a,0))
    if Ll>=a:
    	syst.fill(template_sc_left, shape_left_sc, (-Ll,0))
    if Lr>=a:
    	syst.fill(template_sc_right, shape_right_sc, (Lm,0))

    # LEAD: SLICE OF BULK ALONG X AXIS
    lead = kwant.Builder(kwant.TranslationalSymmetry([0,-a]))

    lead.fill(template_normal, shape_lead(a,Lm-a), (a,0))
    lead.fill(template_barrier, shape_lead(0,a), (0,0))
    lead.fill(template_barrier, shape_lead(Lm-a,Lm), (Lm-a,0))
    if Ll>=a:
        lead.fill(template_sc_left, shape_lead(-Ll,0), (-Ll, 0))
    if Lr>=a:
        lead.fill(template_sc_right, shape_lead(Lm,Lm + Lr), (Lm,0))

    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    
    syst = syst.finalized()

    return syst

@lru_cache()
def make_ns_junction(a, Lm, Lr, Ll, Ly,
                        transverse_soi = True,
                        mu_from_bottom_of_spin_orbit_bands = True):
    """ 
    Builds and returns finalized NS junction system, for calculating transmission
    
    Parameters:
    -----------
    a : float
        lattice spacing in nm

    Lm : float
        width of middle normal strip

    Lr : float
        width of right superconductor

    Ll : float
        width of left superconductor

    Ly : float
        length of finite system

    Returns:
    --------
    syst : kwant.system.FiniteSystem
        Finite system where lead[0] is assumed to be the bulk lead, a slice of the bulk along the y-axis
    """

    #     HAMILTONIAN DEFINITIONS
    template_strings = get_template_strings(transverse_soi, mu_from_bottom_of_spin_orbit_bands)

    # TURN HAMILTONIAN STRINGS INTO TEMPLATES
    template_barrier = kwant.continuum.discretize(template_strings['ham_barrier'], grid_spacing=a)
    template_normal = kwant.continuum.discretize(template_strings['ham_normal'], grid_spacing=a)
    template_sc_left = kwant.continuum.discretize(template_strings['ham_sc_left'], grid_spacing=a)
    template_sc_right = kwant.continuum.discretize(template_strings['ham_sc_right'], grid_spacing=a)
    
    def shape_barrier(site):
        (x, y) = site.pos
        return x==0 and y == 0

    def shape_lead(site):
        (x, y) = site.pos
        return y == 0

    # BUILD SYSTEM
    conservation_matrix = -supercurrent.sigz # Allows seperate bookkeeping of eh in normal lead

    # Make left normal lead
    normal_lead_symmetry = kwant.TranslationalSymmetry((-a,0), (0, a))
    normal_lead = kwant.Builder(normal_lead_symmetry, conservation_law=conservation_matrix)
    normal_lead.fill(template_normal, shape_lead, (0,0))

    # Make right superconducting lead
    sc_lead_symmetry = kwant.TranslationalSymmetry((a,0), (0, a))
    sc_lead = kwant.Builder(sc_lead_symmetry)
    sc_lead.fill(template_sc_right, shape_lead, (a,0))

    # Make barrier/middle site
    wraparound_symmetry = kwant.TranslationalSymmetry((0, a))
    barrier = kwant.Builder(symmetry=wraparound_symmetry, conservation_law=conservation_matrix)
    barrier.fill(template_barrier, shape_barrier, (0,0))

    # Wraparound systems
    barrier = kwant.wraparound.wraparound(barrier, coordinate_names='y') # Specify coordinate name as it otherwise assumes x
    normal_lead = kwant.wraparound.wraparound(normal_lead, keep=0) # Keep lead in x-direction
    sc_lead = kwant.wraparound.wraparound(sc_lead, keep=0) # Keep lead in x-direction

    # Attach leads
    barrier.attach_lead(normal_lead)
    barrier.attach_lead(sc_lead)
    
    return barrier.finalized()

@lru_cache()
def make_wrapped_system(a, Lm, Lr, Ll, Ly,
                        transverse_soi = True,
                        mu_from_bottom_of_spin_orbit_bands = True):

    template_strings = get_template_strings(transverse_soi, mu_from_bottom_of_spin_orbit_bands)

    # TURN HAMILTONIAN STRINGS INTO TEMPLATES
    template_barrier = kwant.continuum.discretize(template_strings['ham_barrier'], grid_spacing=a)
    template_normal = kwant.continuum.discretize(template_strings['ham_normal'], grid_spacing=a)
    template_sc_left = kwant.continuum.discretize(template_strings['ham_sc_left'], grid_spacing=a)
    template_sc_right = kwant.continuum.discretize(template_strings['ham_sc_right'], grid_spacing=a)
    
    # SHAPE FUNCTIONS
    def shape_barrier(site):
        (x, y) = site.pos
        return (0 <= y < Ly) and (x == 0 or x== Lm - a)

    def shape_normal(site):
        (x, y) = site.pos
        return 0 <= y < Ly and a <= x < Lm - a
    
    def shape_left_sc(site):
        (x, y) = site.pos
        return 0 <= y < Ly and -Ll <= x < 0
    
    def shape_right_sc(site):
        (x, y) = site.pos
        return 0 <= y < Ly and Lm <= x < Lm + Lr

    def shape_lead(x1, x2):
        def shape(site):
            (x, y) = site.pos
            return  x1 <= x < x2
        return shape
    
    # BUILD FINITE SYSTEM
    sym = kwant.TranslationalSymmetry((0, a))
    syst = kwant.Builder(symmetry=sym)

    syst.fill(template_normal, shape_normal, (a,0))
    syst.fill(template_barrier, shape_barrier, (0,0))
    syst.fill(template_barrier, shape_barrier, (Lm-a,0))
    if Ll>=a:
        syst.fill(template_sc_left, shape_left_sc, (-Ll,0))
    if Lr>=a:
        syst.fill(template_sc_right, shape_right_sc, (Lm,0))

    lead = kwant.Builder(kwant.TranslationalSymmetry([0,-a]))

    syst = kwant.wraparound.wraparound(syst, coordinate_names='y')
    return syst.finalized()

def to_site_ph_spin(syst_pars, wf):
    norbs = 4
    nsites = len(wf) // norbs
    nsitesL = int(syst_pars['Ly'] / syst_pars['a'])
    nsitesW = nsites // nsitesL

    wf_eh_sp = np.reshape(wf, (nsites, norbs))
    wf_eh_sp_grid = np.reshape(wf_eh_sp, (nsitesW, nsitesL, norbs))
    
    return wf_eh_sp_grid
