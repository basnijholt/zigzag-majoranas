import kwant
import topology
import cmath
import scipy.constants

constants = dict(
    m_eff=0.023 * scipy.constants.m_e / (scipy.constants.eV * 1e-3) / 1e18,  # effective mass in kg, 
    hbar=scipy.constants.hbar / (scipy.constants.eV * 1e-3),
    mu_B=scipy.constants.physical_constants['Bohr magneton'][0] / (scipy.constants.eV * 1e-3),
    exp=cmath.exp,
    cos=cmath.cos,
    sin=cmath.sin)

def make_sns_system(a, Lm, Lr, Ll, Ly, transverse_soi = False):
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
    if transverse_soi:
        ham_str = """
            (hbar^2 / (2*m_eff) * (k_x^2 + k_y^2) - mu) * kron(sigma_z, sigma_0) + 
            g_factor*mu_B*B * kron(sigma_0, sigma_y)"""
    else:
        ham_str = """
        (hbar^2 / (2*m_eff) * (k_x^2 + k_y^2) - mu) * kron(sigma_z, sigma_0) + 
        alpha * (kron(sigma_z, sigma_x) * k_y - kron(sigma_z, sigma_y) * k_x) + 
        g_factor*mu_B*B * kron(sigma_0, sigma_y)"""

    ham_sc_left = ham_str + " + Delta * kron(sigma_y, sigma_0)"
    ham_sc_right = ham_str + """ + cos(phase) * Delta * kron(sigma_y, sigma_0) + 
                                    + sin(phase) * Delta * kron(sigma_x, sigma_0)
    """


    # TURN HAMILTONIAN STRINGS INTO TEMPLATES
    template_normal = kwant.continuum.discretize(ham_str, grid_spacing=a)
    template_sc_left = kwant.continuum.discretize(ham_sc_left, grid_spacing=a)
    template_sc_right = kwant.continuum.discretize(ham_sc_right, grid_spacing=a)
    
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
    syst.fill(template_sc_left, shape_left_sc, (-Ll,0))
    syst.fill(template_sc_right, shape_right_sc, (Lm,0))

    # LEAD: SLICE OF BULK ALONG Y AXIS
    lead = kwant.Builder(kwant.TranslationalSymmetry([0,-a]))

    lead.fill(template_sc_left, shape_lead(-Ll,0), (-Ll, -a))
    lead.fill(template_normal, shape_lead(0,Lm), (0,-a))
    lead.fill(template_sc_right, shape_lead(Lm,Lm + Lr), (Lm,-a))

    syst.attach_lead(lead)
    
    return syst.finalized()

