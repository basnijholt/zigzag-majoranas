import itertools
import copy
import collections
from functools import partial
from itertools import product
import numpy as np
import sns_system, topology, spectrum, supercurrent
import adaptive
import holoviews as hv

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


def pfaffian_function(_, syst_total, syst_wrapped, syst_junction, syst_pars, params):
    return topology.get_pfaffian(syst_total, params)


def current_function(current_params, syst_total, syst_wrapped, syst_junction, syst_pars, params):
    return supercurrent.wrapped_current(syst_pars=syst_pars,
                                 params=params,
                                 syst_wrapped=syst_wrapped,
                                 **current_params)


def energy_gap_function(energy_gap_params, syst_total, syst_wrapped, syst_junction, syst_pars, params):
    return spectrum.find_gap_of_lead(lead=syst_total.leads[0],
                              params=params,
                              **energy_gap_params)


def transparency_function(self, transparency_params):
    return None

def bandstructure_function(self, bandstructure_params):
    return None

def get_correct_metric_function(metric_key, metric_params):
    options = {
        'pfaffian': pfaffian_function,
        'current': current_function,
        'transparency': transparency_function,
        'bandstructure': bandstructure_function,
        'energy_gap': energy_gap_function,
    }
    return partial(options[metric_key], metric_params)

def total_function(xy, syst_pars, params, keys_with_bounds, metric_params_dict):
    syst_total    = sns_system.make_sns_system(**syst_pars)
    syst_wrapped  = sns_system.make_wrapped_system(**syst_pars)
    syst_junction = sns_system.make_junction_system(**syst_pars)

    results = np.zeros(len(metric_params_dict))

    keys = keys_with_bounds.keys()
    params_local = params.copy()
    for k, val in zip(keys_with_bounds, xy):
        params_local[k] = val
    
    for idx, (metric_key, metric_params) in enumerate(metric_params_dict.items()):
        metric_function = get_correct_metric_function(metric_key, metric_params)
        results[idx] = metric_function(syst_total,
                                       syst_wrapped,
                                       syst_junction,
                                       syst_pars,
                                       params_local)
        
    return results

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

        self.metric_params_dict = metric_params_dict

    @property
    def xscale(self):
        return abs(self.bounds[0][0]-self.bounds[0][1])
    
    @property
    def yscale(self):
        return abs(self.bounds[1][0]-self.bounds[1][1])

    @property
    def xcenter(self):
        return (self.bounds[0][0]+self.bounds[0][1])/2
   
    @property
    def ycenter(self):
        return (self.bounds[1][0]+self.bounds[1][1])/2

    def unnormalize(self, x, y):
        x_unscaled = self.xscale * x + self.xcenter
        y_unscaled = self.yscale * y + self.ycenter
        return(x_unscaled, y_unscaled)

    def get_total_function(self):
        return partial(total_function,
                       syst_pars=self.syst_pars,
                       params=self.params,
                       keys_with_bounds=self.keys_with_bounds,
                       metric_params_dict=self.metric_params_dict)

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
        self.learner = adaptive.Learner2D(f, bounds=self.bounds, loss_per_triangle=self.default_metric())
        return self.learner

    def plot(self, n=200, pfaffian_contour=True):
        plot_dictionary = self.get_plot_dictionary(n)
        map_plot = hv.HoloMap(plot_dictionary,
                              kdims='Metric').opts(norm=dict(framewise=True))

        keys = list(self.keys_with_bounds.keys())

        if pfaffian_contour:
            xy = get_pfaffian_contour(plot_dictionary)
            contour_pfaffian = hv.Path(xy)
            return (map_plot*contour_pfaffian).redim(x=keys[0], y=keys[1])
        else:
            return map_plot.redim(x=keys[0], y=keys[1])

    def get_plot_dictionary(self, n):
        ip = self.learner.ip()
        normalized_dim = np.linspace(-.5,.5, n)
        xdim, ydim = self.unnormalize(normalized_dim, normalized_dim)

        gridx, gridy = np.meshgrid(normalized_dim, normalized_dim)
        gridded_results = ip(gridx, gridy)

        plot_dictionary = dict()
        for idx, metric_key in enumerate(self.metric_params_dict):
            metric_results = gridded_results[:,:, idx]
            plot_dictionary[metric_key] = hv.Image((xdim, ydim, metric_results))

        return plot_dictionary

def get_pfaffian_contour(plot_dictionary):
        contour_pfaffian = hv.operation.contours(plot_dictionary["pfaffian"], levels=0)
        xdata = contour_pfaffian.data[0]['x']
        ydata = contour_pfaffian.data[0]['y']
        return (xdata, ydata)

class AggregatesSimulationSet:
    def __init__(self,
                 keys_with_bounds,
                 syst_pars,
                 params,
                 metric_params_dict):

        self.syst_pars_dimensions = {}
        self.params_dimensions = {}
        

        self.keys_with_bounds = keys_with_bounds.copy()
        
        self.bounds = list(keys_with_bounds.values())

        self.syst_pars = syst_pars.copy()
        self.params    = params.copy()
        self.params.update(dict(**sns_system.constants))

        self.metric_params_dict = metric_params_dict

    
    def add_dimension(self, dimension_name, dimension_values):
        if self.syst_pars.get(dimension_name):
            self.syst_pars_dimensions[dimension_name] = dimension_values
        elif self.params.get(dimension_name):
            self.params_dimensions[dimension_name] = dimension_values
        else:
            raise KeyError(dimension_name + " is not a parameter in syst_pars or params")

    def make_simulation_sets(self):
        simulation_set_list = []
        for parameter_values in product(*self.syst_pars_dimensions.values()):
            update_dict = zip(self.syst_pars_dimensions.keys(), parameter_values)
            syst_pars = self.syst_pars.update(update_dict)
            for parameter_values in product(*self.params_dimensions.values()):
                update_dict = zip(self.params_dimensions.keys(), parameter_values)
                params = self.params.update(update_dict)
                ss = SimulationSet(self.keys_with_bounds,
                                   self.syst_pars,
                                   self.params,
                                   self.metric_params_dict)
                simulation_set_list.append(ss)

        self.simulation_set_list = simulation_set_list

    def make_learners(self):
        self.make_simulation_sets()
        self.learners = [ss.get_learner() for ss in self.simulation_set_list]

    def get_balancing_learner(self):
        self.make_learners()
        return adaptive.BalancingLearner(self.learners)

