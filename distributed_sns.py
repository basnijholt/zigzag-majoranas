import itertools
import copy
import collections
from functools import partial
import numpy as np
import sns_system, topology, spectrum, supercurrent
import adaptive

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

class SimulationSet():
    def __init__(self,
                 keys_with_bounds,
                 syst_pars,
                 params,
                 metric_params_dict):

        self.keys_with_bounds = keys_with_bounds.copy()
        
        self.bounds = list(keys_with_bounds.values())

        self.syst_pars = syst_pars.copy()
        self.params    = params.copy()
        self.params.update(dict(**sns_system.constants))

        self.metric_index_dict = {metric_key:idx for (idx, metric_key) in enumerate(metric_params_dict.keys())}
        self.metric_params_dict = metric_params_dict

    def get_correct_metric_function(self, metric_key):
        metric_params = self.metric_params_dict[metric_key]
        if metric_key=="pfaffian":
            return self.get_pfaffian_function(metric_params)
        elif metric_key=="current":
            return self.get_current_function(metric_params)
        elif metric_key=="transparency":
            return self.get_transparency_function(metric_params)
        elif metric_key=="bandstructure":
            return self.get_bandstructure_function(metric_params)
        elif metric_key=="energy_gap":
            return self.get_energy_gap_function(metric_params)
        else:
            raise ValueError(metric_key + " is not a valid metric")

    def get_pfaffian_function(self, pfaffian_params):
        def pfaffian_function(syst_total, syst_wrapped, syst_junction, syst_pars, params):
            return topology.get_pfaffian(syst_total, params)

        return pfaffian_function

    def get_current_function(self, current_params):
        def current_function(syst_total, syst_wrapped, syst_junction, syst_pars, params):
            return supercurrent.wrapped_current(syst_pars=syst_pars,
                                         params=params,
                                         syst_wrapped=syst_wrapped,
                                         **current_params)

        return current_function

    def get_energy_gap_function(self, energy_gap_params):
        def energy_gap_function(syst_total, syst_wrapped, syst_junction, syst_pars, params):
            return spectrum.find_gap_of_lead(lead=syst_total.leads[0],
                                      params=params,
                                      **energy_gap_params)

        return energy_gap_function

    def get_transparency_function(self, transparency_params):
        return None

    def get_bandstructure_function(self, bandstructure_params):
        return None

    def get_total_function(self):
        function_list = list()
        for metric_key, metric_idx in self.metric_index_dict.items():
            f = self.get_correct_metric_function(metric_key)
            function_list.append(f)

        def total_function(xy, syst_pars, params):
            syst_total    = sns_system.make_sns_system(**syst_pars)
            syst_wrapped  = sns_system.make_wrapped_system(**syst_pars)
            syst_junction = sns_system.make_junction_system(**syst_pars)

            results = np.zeros(len(self.metric_index_dict))
            
            keys = self.keys_with_bounds.keys()
            params_local = params.copy()
            for k, val in zip(self.keys_with_bounds, xy):
                params_local[k] = val
            
            for idx, metric_function in enumerate(function_list):
                results[idx] = metric_function(syst_total, syst_wrapped, syst_junction, syst_pars, params_local)

            return results

        return partial(total_function, syst_pars=self.syst_pars, params=self.params)

    def default_metric(self):
        def default_loss_scaled(ip):
            from adaptive.learner.learner2D import deviations, areas
            metric_scale = np.max(ip.values, 0)-np.min(ip.values, 0)
            metric_scale[metric_scale==0] = 1
            
            dev = np.sum(deviations(ip)/metric_scale[:,np.newaxis], axis=0)
            A = areas(ip)
            losses = dev * np.sqrt(A) + 0.3 * A
            return losses

        return default_loss_scaled

    def get_learner(self):
        f = self.get_total_function()
        learner = adaptive.Learner2D(f, bounds=self.bounds, loss_per_triangle=self.default_metric())
        return learner
