import kwant
import sns_system


def transparency(syst_junction, params, k_x=0):
    params['k_x'] = k_x
    smatrix = kwant.smatrix(syst_junction, energy=0, params=params)
    # spin accounts for double transmission
    return 1 - smatrix.transmission((0, 0), (0, 0)) / 2
