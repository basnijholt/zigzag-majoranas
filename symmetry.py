import numpy as np


def is_phs(ham):
    """
    Checks wheteher matrix is indeed particle hole symmetric,
    assuming we are in the  {Ψ_e↑, Ψ_e↓, Ψ_h↓, -Ψ_h↑} basis.

    Parameters
    ----------
            ham : numpy.ndarray

    Returns
    -------
            ham == - ham(holes <-> electrons) : bool
    """

    sym_mat = np.kron(np.eye(ham.shape[0] // 4), np.kron(sigma_y, sigma_y))
    ham_phhp = sym_mat @ ham.conj() @ sym_mat
    return (np.abs(ham + ham_phhp) < 1e-12).all()


def is_antisymmetric(ham):
    """ Checks whether matrix is antisymmetric.

    Parameters
    ----------
            ham : numpy.ndarray

    Returns
    -------
            ham == - ham.transposed : bool
    """
    return (np.abs(ham + ham.T) < 1e-12).all()


def make_skew_symmetric(ham):
    """
    From Bas Nijholt's code:
    Makes a skew symmetric matrix by a matrix multiplication of a unitary
    matrix U. This unitary matrix is taken from the Topology MOOC 0D, but
    that is in a different basis. To get to the right basis one multiplies
    by [[np.eye(2), 0], [0, sigma_y]].

    Parameters
    ----------
    ham : numpy.ndarray
        Hamiltonian matrix gotten from sys.cell_hamiltonian()

    Returns
    -------
    skew_ham : numpy.ndarray
        Skew symmetrized Hamiltonian
    """
    W = ham.shape[0] // 4
    I = np.eye(2, dtype=complex)
    sigma_y = np.array([[0, 1j], [-1j, 0]], dtype=complex)
    U_1 = np.bmat([[I, I], [1j * I, -1j * I]])
    U_2 = np.bmat([[I, 0 * I], [0 * I, sigma_y]])
    U = U_1 @ U_2
    U = np.kron(np.eye(W, dtype=complex), U)
    skew_ham = U @ ham @ U.H

    assert is_antisymmetric(skew_ham)

    return skew_ham
