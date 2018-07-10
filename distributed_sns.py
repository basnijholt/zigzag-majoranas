import itertools
import copy
import collections

def f_adaptive(xy, keys, params, syst_pars,
                     transverse_soi=True,
                     zeeman_in_superconductor=False):
    import sns_system, topology

    params.update(dict(**sns_system.constants))
    
    for k, val in zip(keys, xy):
        params[k] = val
    params[keys[0]], params[keys[1]] = xy
    syst = sns_system.make_sns_system(**syst_pars, transverse_soi=transverse_soi, zeeman_in_superconductor=zeeman_in_superconductor)
    return topology.get_pfaffian(syst, params)


def get_list_of_parameter_dictionaries(params_mutable):
    """ 
    Turn dictionary containing parameters -- where some of the parameters 
    are arrays of parameters -- into a list of dictionaries, containing single
    element parameters.
    
    Parameters:
    -----------
    params_mutable : dict
        dictionary containing parameter (-ranges)

    Returns:
    --------
    iterable_keys : list
        List of keys of parameters that contained multiple entries

    param_list : list
        List of parameter dictionaries
    """

    params = copy.copy(params_mutable)
    
    iterable_keys = []
    iterable_vals = []
    param_list = []

    for k,v in params.items():
        if isinstance(v, collections.Iterable):
            iterable_keys.append(k)
            iterable_vals.append(copy.copy(v))

    for k in iterable_keys:
        params[k] = 0

    for vals in itertools.product(*iterable_vals):
        iter_param_dict = dict(zip(iterable_keys, vals))
        param_element = copy.copy(params)
        param_element.update(iter_param_dict)
        param_list.append(param_element)

    return (iterable_keys, param_list)