import numpy as np
import scipy.linalg as la
import numpy as np

sigma_0 = np.array([[1, 0], [0, 1]], dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)


def symmetry(H, sigma, tau, norbs=4):
    W = H.shape[0] // norbs
    U = np.kron(sigma, tau)
    U = np.kron(np.identity(W), U)
    return U


def phs(H):
    return symmetry(H, sigma_y, sigma_y, 4)


def phs_symmetrization(wfs, particle_hole):
    """Makes the wave functions that have the same velocity at a time-reversal
    invariant momentum (TRIM) particle-hole symmetric.
    If P is the particle-hole operator and P^2 = 1, then a particle-hole
    symmetric wave function at a TRIM is an eigenstate of P with eigenvalue 1.
    If P^2 = -1, wave functions with the same velocity at a TRIM come in pairs.
    Such a pair is particle-hole symmetric if the wave functions are related by
    P, i. e. the pair can be expressed as [psi_n, P psi_n] where psi_n is a wave
    function.
    To ensure proper ordering of modes, this function also returns an array
    of indices which ensures that particle-hole partners are properly ordered
    in this subspace of modes. These are later used with np.lexsort to ensure
    proper ordering.
    Parameters
    ----------
    wfs : numpy array
        A matrix of propagating wave functions at a TRIM that all have the same
        velocity. The wave functions form the columns of this matrix.
    particle_hole : numpy array
        The matrix representation of the unitary part of the particle-hole
        operator, expressed in the tight binding basis.
    Returns
    -------
    new_wfs : numpy array
        The matrix of particle-hole symmetric wave functions.
    TRIM_sort: numpy integer array
        Index array that stores the proper sort order of particle-hole symmetric
        wave functions in this subspace.
    """
    def Pdot(mat):
        """Apply the particle-hole operator to an array. """
        return particle_hole @ mat.conj()
    # P always squares to 1 or -1.
    P_squared = np.sign(particle_hole[0, :] @ particle_hole[:, 0].conj())
    # np.sign returns the same data type as its argument. Make sure
    # that the comparison with integers is okay.
    assert P_squared in (-1, 1)
    # Check that wf are linearly independent and the space spanned
    # by them is closed under ph
    if np.isclose(la.det(wfs.T.conj() @ Pdot(wfs)), 0):
        raise ValueError(
            'wfs are not indepentent or not closed under particle_hole.')
    if P_squared == 1:
        new_wfs = np.empty((wfs.shape[0], 0))
        while wfs.shape[0] > 0:
            # Make particle hole eigenstates.
            # Phase factor ensures they are not numerically close and no vector
            # in ph_wfs is close to zero.
            phases = np.diag([np.exp(0.5j*np.angle(wf.T.conj() @ Pdot(wf)))
                              for wf in wfs.T])
            ph_wfs = wfs @ phases + Pdot(wfs @ phases)
            # Orthonormalize the modes using QR on the matrix of eigenstates of P.
            # So long as the matrix of coefficients R is purely real, any linear
            # combination of these modes remains an eigenstate of P. From the way
            # we construct eigenstates of P, the coefficients of R are real.
            ph_wfs = la.qr(ph_wfs, mode='economic', pivoting=True)[0]
            # If one wf is ph image of another, there may be some duplicates in
            # the original ph_wfs and it does not span the same space as wfs, so some
            # vectors produced by QR may not be ph eigenstates.
            # Find those that are not ph eigenstates.
            null_space = np.array([not np.allclose(wf, Pdot(wf))
                                   for wf in ph_wfs.T])
            # Add the good ph_wfs to the result
            new_wfs = np.hstack((new_wfs, ph_wfs[:, np.invert(null_space)]))
            # If there were no bad vectors, we are done
            if sum(null_space) == 0:
                break
            # project wave functions onto the remaining subspace
            projector = ph_wfs[:, null_space] @ ph_wfs[:, null_space].T.conj()
            wfs = projector @ wfs
            # Find the a basis spanning the remaining subspace and start again
            # with only these vectors.
            wfs = la.qr(wfs, mode='economic', pivoting=True)[0]
            wfs = wfs[:, :sum(null_space)]
        # If P^2 = 1, there is no need to sort the modes further.
        TRIM_sort = np.zeros((wfs.shape[1],), dtype=int)
    else:
        # P^2 = -1.
        # Iterate over wave functions to construct
        # particle-hole partners.
        new_wfs = []
        # The number of modes. This is always an even number >=2.
        N_modes = wfs.shape[1]
        # If there are only two modes in this subspace, they are orthogonal
        # so we replace the second one with the P applied to the first one.
        if N_modes == 2:
            wf = wfs[:, 0]
            # Store psi_n and P psi_n.
            new_wfs.append(wf)
            new_wfs.append(Pdot(wf))
        # If there are more than two modes, iterate over wave functions
        # and construct their particle-hole partners one by one.
        else:
            # We construct pairs of modes that are particle-hole partners.
            # Need to iterate over all pairs except the final one.
            iterations = range((N_modes-2)//2)
            for i in iterations:
                # Take a mode psi_n from the basis - the first column
                # of the matrix of remaining modes.
                wf = wfs[:, 0]
                # Store psi_n and P psi_n.
                new_wfs.append(wf)
                P_wf = Pdot(wf)
                new_wfs.append(P_wf)
                # Remove psi_n and P psi_n from the basis matrix of modes.
                # First remove psi_n.
                wfs = wfs[:, 1:]
                # Now we project the remaining modes onto the orthogonal
                # complement of P psi_n. projector:
                projector = (wfs @ wfs.T.conj() -
                             np.outer(P_wf, P_wf.T.conj()))
                # After the projection, the mode matrix is rank deficient -
                # the span of the column space has dimension one less than
                # the number of columns.
                wfs = projector @ wfs
                wfs = la.qr(wfs, mode='economic', pivoting=True)[0]
                # Remove the redundant column.
                wfs = wfs[:, :-1]
                # If this is the final iteration, we only have two modes
                # left and can construct particle-hole partners without
                # the projection.
                if i == iterations[-1]:
                    assert wfs.shape[1] == 2
                    wf = wfs[:, 0]
                    # Store psi_n and P psi_n.
                    new_wfs.append(wf)
                    new_wfs.append(Pdot(wf))
                assert np.allclose(wfs.T.conj() @ wfs,
                                   np.eye(wfs.shape[1]))
        new_wfs = np.hstack([col.reshape(len(col), 1)/npl.norm(col) for
                             col in new_wfs])
        assert np.allclose(new_wfs[:, 1::2], Pdot(new_wfs[:, ::2]))
        # Store sort ordering in this subspace of modes
        TRIM_sort = np.arange(new_wfs.shape[1])
    assert np.allclose(new_wfs.T.conj() @ new_wfs, np.eye(new_wfs.shape[1]))
    return new_wfs, TRIM_sort


def is_phs(ham):
	""" 
	Checks wheteher matrix is indeed particle hole symmetric, 
	assuming we are in the  {Ψ_e↑, Ψ_e↓, Ψ_h↓, -Ψ_h↑} basis.

	Parameters:
	-----------
		ham : numpy.ndarray

	Returns:
	--------
		ham == - ham(holes <-> electrons) : bool
	"""
	
	sym_mat = np.kron(np.eye(ham.shape[0]//4), np.kron(sigma_y, sigma_y)) 
	ham_phhp = sym_mat @ ham.conj() @ sym_mat
	return (np.abs(ham + ham_phhp) < 1e-12).all()

def is_antisymmetric(ham):
	""" Checks whether matrix is antisymmetric.
	
	Parameters:
	-----------
		ham : numpy.ndarray

	Returns:
	--------
		ham == - ham.transposed : bool
	"""
	return (np.abs(ham + ham.T) < 1e-12).all()

def make_skew_symmetric(ham):
    """
    Turns hamiltonian in arbitrary basis into skew symmetric matrix

    Parameters:
    -----------
    ham : numpy.ndarray
        Hamiltonian matrix gotten from sys.cell_hamiltonian()

    Returns:
    --------
    skew_ham : numpy.ndarray
        Skew symmetrized Hamiltonian
    """

    I = np.eye(ham.shape[0], dtype=complex)
    U = phs_symmetrization(I, phs(I))[0]
    skew_ham =  U.T.conj() @ ham @ U
    assert is_antisymmetric(skew_ham)

    return skew_ham

