import kwant
import numpy as np
import symmetry
import dependencies.pfaffian as pfaffian
import cmath

def get_bulk_hamiltonian(syst, params):
	""" 
	Returns H(k, y) of infinitely extended system in x direction.
	
	Parameters:
	-----------
	syst : kwant.system.FiniteSystem
		Finite system where lead[0] is assumed to be the bulk lead, a slice of the bulk along the y-axis

	params : dict
		Set of parameters for the system

	Returns:
	--------
	H(k) : function returning np.ndarray
		Returns a function which returns the Hamiltonian H(k_y)
	"""
	bulk_system = syst.leads[0]
	
	h = bulk_system.cell_hamiltonian(params = params)
	t = bulk_system.inter_cell_hopping(params = params)
	
	return lambda k: h + t*np.exp(1j*k) + t.T.conj()*np.exp(-1j*k)


def get_pfaffian(syst, params):
	""" 
	Returns pfaffian of a system
	
	Parameters:
	-----------
	syst : kwant.system.FiniteSystem
		Finite system where lead[0] is assumed to be the bulk lead, a slice of the bulk along the y-axis

	params : dict
		Set of parameters for the system

	Returns:
	--------
		sign(Pf(H(0))) * sign(Pf(H(pi))) : float {-1, +1}
	"""
	h_k = get_bulk_hamiltonian(syst, params)

	h_0  = h_k(0)
	h_pi = h_k(np.pi)

	skew_pf_0 = symmetry.make_skew_symmetric(h_0)
	skew_pf_pi = symmetry.make_skew_symmetric(h_pi)
	
	pf_0  = pfaffian.pfaffian(1j*skew_pf_0, sign_only=True).real
	pf_pi = pfaffian.pfaffian(1j*skew_pf_pi, sign_only=True).real
	return pf_0*pf_pi