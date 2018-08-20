import kwant
import sns_system

def transparency(syst_junction, params, k_y=0):
    params['k_y'] = k_y
    smatrix = kwant.smatrix(syst_junction, energy=0, params=params)
    return 1 - smatrix.transmission((0,0),(0,0))/2 # spin accounts for double transmission



