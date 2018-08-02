import kwant
import topology
import cmath
import scipy.constants
import numpy as np
import supercurrent

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
                        T=12)

dummy_params = dict(**constants,
                    **dummy_params_raw)

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
        ham_normal += """
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

    ham_sc_left  += "+ g_factor_left * mu_B * B * kron(sigma_0, sigma_y)"
    ham_sc_right += "+ g_factor_right * mu_B * B * kron(sigma_0, sigma_y)"
        
    template_strings = dict(ham_normal=ham_normal,
                            ham_sc_right=ham_sc_right,
                            ham_sc_left=ham_sc_left)
    return template_strings

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
    template_normal = kwant.continuum.discretize(template_strings['ham_normal'], grid_spacing=a)
    template_sc_left = kwant.continuum.discretize(template_strings['ham_sc_left'], grid_spacing=a)
    template_sc_right = kwant.continuum.discretize(template_strings['ham_sc_right'], grid_spacing=a)
    
    # SHAPE FUNCTIONS
    def shape_normal(site):
        (x, y) = site.pos
        return 0 <= y < Ly and 0 <= x < Lm
    
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

    syst.fill(template_normal, shape_normal, (0,0))
    if Ll>=a:
    	syst.fill(template_sc_left, shape_left_sc, (-Ll,0))
    if Lr>=a:
    	syst.fill(template_sc_right, shape_right_sc, (Lm,0))

    # LEAD: SLICE OF BULK ALONG Y AXIS
    lead = kwant.Builder(kwant.TranslationalSymmetry([0,-a]))

    lead.fill(template_normal, shape_lead(0,Lm), (0,-a))
    if Ll>=a:
        lead.fill(template_sc_left, shape_lead(-Ll,0), (-Ll, -a))
    if Lr>=a:
        lead.fill(template_sc_right, shape_lead(Lm,Lm + Lr), (Lm,-a))

    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    
    syst = syst.finalized()

    return syst

def make_junction_system(a, Lm, Lr, Ll, Ly,
                         transverse_soi = True,
                         mu_from_bottom_of_spin_orbit_bands = True):
    """ 
    Builds and returns finalized junction of the sns system
    
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
    template_normal = kwant.continuum.discretize(template_strings['ham_normal'], grid_spacing=a)
    template_sc_left = kwant.continuum.discretize(template_strings['ham_sc_left'], grid_spacing=a)
    template_sc_right = kwant.continuum.discretize(template_strings['ham_sc_right'], grid_spacing=a)
    
    # SHAPE FUNCTIONS
    def shape_normal(site):
        (x, y) = site.pos
        return 0 <= y < Ly and 0 <= x < Lm
    
    def shape_left_sc(site):
        (x, y) = site.pos
        return 0 <= y < Ly
    
    def shape_right_sc(site):
        (x, y) = site.pos
        return 0 <= y < Ly
    
    # BUILD FINITE SYSTEM
    syst = kwant.Builder()
    syst.fill(template_normal, shape_normal, (0,0))

    lead_left = kwant.Builder(kwant.TranslationalSymmetry([-a, 0]))
    lead_right = kwant.Builder(kwant.TranslationalSymmetry([a, 0]))

    lead_left.fill(template_sc_left, shape_left_sc, (-a, 0))
    lead_right.fill(template_sc_right, shape_right_sc, (Lm, 0))

    syst.attach_lead(lead_left)
    syst.attach_lead(lead_right)
    
    return syst.finalized()

def make_wrapped_system(a, Lm, Lr, Ll, Ly,
                        transverse_soi = True,
                        mu_from_bottom_of_spin_orbit_bands = True):
    template_strings = get_template_strings(transverse_soi, mu_from_bottom_of_spin_orbit_bands)

    # TURN HAMILTONIAN STRINGS INTO TEMPLATES
    template_normal = kwant.continuum.discretize(template_strings['ham_normal'], grid_spacing=a)
    template_sc_left = kwant.continuum.discretize(template_strings['ham_sc_left'], grid_spacing=a)
    template_sc_right = kwant.continuum.discretize(template_strings['ham_sc_right'], grid_spacing=a)

    # SHAPE FUNCTIONS
    def shape_normal(site):
        (x, y) = site.pos
        return 0 <= x < Lm

    def shape_left_sc(site):
        (x, y) = site.pos
        return -Ll <= x < 0

    def shape_right_sc(site):
        (x, y) = site.pos
        return Lm <= x < Lm + Lr
    
    def shape_lead(x1, x2):
        def shape(site):
            (x, y) = site.pos
            return  x1 <= x < x2
        return shape

    # BUILD FINITE SYSTEM
    sym = kwant.TranslationalSymmetry((0, a))
    syst = kwant.Builder(symmetry=sym)

    syst.fill(template_normal, shape_normal, (0,0))
    if Ll>=a:
        syst.fill(template_sc_left, shape_left_sc, (-Ll,0))
    if Lr>=a:
        syst.fill(template_sc_right, shape_right_sc, (Lm,0))

    syst = kwant.wraparound.wraparound(syst)
    return syst.finalized()

def to_site_ph_spin(syst_pars, wf):
    norbs = 4
    nsites = len(wf) // norbs
    nsitesL = int(syst_pars['Ly'] / syst_pars['a'])
    nsitesW = nsites // nsitesL

    wf_eh_sp = np.reshape(wf, (nsites, norbs))
    wf_eh_sp_grid = np.reshape(wf_eh_sp, (nsitesW, nsitesL, norbs))
    
    return wf_eh_sp_grid
