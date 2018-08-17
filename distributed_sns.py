import itertools
import copy
import collections
from functools import partial
from itertools import product
import numpy as np
import sns_system, topology, spectrum, supercurrent, scattering
import adaptive
import dependencies.adaptive_tools as adaptive_tools
import holoviews as hv
import os
import pickle
import dill
from utils import consistent_hash

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


def transparency_function(transparency_params, syst_total, syst_wrapped, syst_junction, syst_pars, params):
    return scattering.transparency(syst_junction, params, **transparency_params)

def bandstructure_function(bandstructure_params, syst_total, syst_wrapped, syst_junction, syst_pars, params):
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
    syst_junction = sns_system.make_ns_junction(**syst_pars)

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
    learner_file_prefix = 'learner_data_'
    simulation_set_file_prefix = 'simulation_set_data_'

    def __init__(self,
                 keys_with_bounds,
                 syst_pars,
                 params,
                 metric_params_dict):

        
        self.keys_with_bounds = keys_with_bounds.copy()
        self.syst_pars = syst_pars.copy()
        self.params    = params.copy()
        self.metric_params_dict = metric_params_dict
        
        
        self.bounds = list(self.keys_with_bounds.values())
        self.params.update(dict(**sns_system.constants))
        self.make_learner()

    @property
    def input_params(self):
        return (self.keys_with_bounds,
                self.syst_pars,
                self.params,
                self.metric_params_dict)
    
    @property
    def table_dicts(self):
        params = {k:v for k,v in self.params.items() if ((k in sns_system.dummy_params_raw) and (k not in self.keys_with_bounds))}
        return (self.syst_pars, params)

    @property
    def hash_str(self):
        n_digits=6
        hash_dict = {}
        hash_func = lambda dictionary: hex(consistent_hash(dictionary) % 2**(4*n_digits))[2:]
        hash_str = '<KB_' + hash_func(self.keys_with_bounds) + '_>'
        hash_str += '<PRMS_' + hash_func([(k, v) for k,v in self.params.items() if type(v) is float or type(v) is int]) + '_>'
        hash_str += '<PARS_' + hash_func([(k, v) for k,v in self.syst_pars.items() if type(v) is float or type(v) is int]) + '_>'
        hash_str += '<METR_' + hash_func([(k, tuple(v.items())) for k,v in self.metric_params_dict.items()]) + '_>'
        return hash_str
    
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

    def make_learner(self):
        f = self.get_total_function()
        self.learner = adaptive_tools.Learner2D(f, bounds=self.bounds, loss_per_triangle=self.default_metric())

    def get_learner(self):
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
    
    def save(self, folder):
        fname=self.hash_str

        self.learner.save(folder, self.learner_file_prefix + fname)
        
        os.makedirs(folder, exist_ok=True)
        fname = os.path.join(folder, self.simulation_set_file_prefix + fname)
        with open(fname, 'wb') as f:
            pickle.dump(self.input_params, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, folder=''):
        self.learner.load(folder, self.learner_file_prefix + self.hash_str)

    def load_from_file(fname, folder=''):
        fname = os.path.join(folder, SimulationSet.simulation_set_file_prefix + fname)

        with open(fname, 'rb') as f:
                input_params = pickle.load(f)

        ss = SimulationSet(*input_params)
        ss.load(folder)
        return ss

def get_pfaffian_contour(plot_dictionary):
        contour_pfaffian = hv.operation.contours(plot_dictionary["pfaffian"], levels=0)
        xdata = contour_pfaffian.data[0]['x']
        ydata = contour_pfaffian.data[0]['y']
        return (xdata, ydata)

def loss_enough_points(loss_function, enough_points):
    def loss_f(ip):
        from adaptive.learner.learner2D import areas
        loss = loss_function(ip)
        dim = areas(ip).shape[0]
        return 1e30 * loss / dim if dim < enough_points else loss
    return loss_f

def ptable(dictionary):
    return hv.Table({'Parameter name':list(dictionary.keys()),
          'Value':list(dictionary.values())},['Parameter name', 'Value'])


class AggregatesSimulationSet():
    learner_file_prefix = 'learner_data_'
    aggregate_simulation_set_file_prefix = 'aggregate_simulation_set_data_'

    def __init__(self,
                 keys_with_bounds,
                 syst_pars,
                 params,
                 metric_params_dict):


        self.dimension_dict = {}
        

        self.keys_with_bounds = keys_with_bounds.copy()
        self.keys = tuple(keys_with_bounds.keys())
        self.bounds = tuple(keys_with_bounds.values())

        self.syst_pars = syst_pars.copy()
        self.params    = params.copy()
        self.params.update(dict(**sns_system.constants))

        self.metric_params_dict = metric_params_dict

    @property
    def input_params(self):
        return (self.keys_with_bounds,
                self.syst_pars,
                self.params,
                self.metric_params_dict)   

    @property
    def learners(self):
        return self.learner.learners
    
    @property
    def simulation_set_dict(self):
            return {tuple(ss.dimension_values):ss for ss in self.simulation_set_list}

    @property
    def hash_str(self):
        n_digits=6
        hash_dict = {}
        hash_func = lambda dictionary: hex(consistent_hash(dictionary) % 2**(4*n_digits))[2:]
        hash_str = '<KB_' + hash_func(self.keys_with_bounds) + '_>'
        hash_str += '<PRMS_' + hash_func(self.params) + '_>'
        hash_str += '<PARS_' + hash_func(self.syst_pars) + '_>'
        hash_str += '<METR_' + hash_func(self.metric_params_dict) + '_>'
        hash_str += '<DIM_' + hash_func(self.dimension_dict) + '_>'
        return hash_str

    def add_dimension(self, dimension_name, dimension_functions):
        self.dimension_dict[dimension_name] = dimension_functions

    def make_simulation_sets(self):
        simulation_set_list = []

        for parameter_altering_functions in product(*self.dimension_dict.values()):
            syst_pars = self.syst_pars.copy()
            params = self.params.copy()
            dimension_values = []

            for function in parameter_altering_functions:
                dimension_values.append(function(syst_pars, params))

            ss = SimulationSet(self.keys_with_bounds,
                               syst_pars,
                               params,
                               self.metric_params_dict)
            ss.dimension_values = dimension_values
            simulation_set_list.append(ss)

        self.simulation_set_list = simulation_set_list

    def make_learners(self, enough_points):
        self.make_simulation_sets()
        for ss in self.simulation_set_list:
            ss.make_learner()
            
        learners = [ss.get_learner() for ss in self.simulation_set_list]
        
        for learner in learners:
            loss_function = learner.loss_per_triangle
            learner.loss_per_triangle = loss_enough_points(loss_function, enough_points)
        return learners

    def make_balancing_learner(self, enough_points=1):
        learners = self.make_learners(enough_points)
        self.learner = adaptive_tools.BalancingLearner(learners)
    
    def get_balancing_learner(self):
        return self.learner

    def get_parameter_table_dicts(self):
        table_dict_params = {}
        table_dict_syst_pars = {}
        for ss_key, ss in self.simulation_set_dict.items():
            for metric_key in self.metric_params_dict:
                syst_pars, params_raw = ss.table_dicts
                table_dict_params[(*ss_key, metric_key)] = ptable(params_raw)
                table_dict_syst_pars[(*ss_key, metric_key)] = ptable(syst_pars)
        return (table_dict_syst_pars, table_dict_params)

    def get_plot_dict(self, n, contour_pfaffian=False, tables=False):
        plot_dict = dict()
        kdims = list(self.dimension_dict.keys()) + ['Metric']
        if tables is not False:
            syst_tables, parameter_tables = self.get_parameter_table_dicts()

        for ss in self.simulation_set_list:
            local_plot_dict = ss.get_plot_dictionary(n)
            if contour_pfaffian is not False:
                xy = get_pfaffian_contour(local_plot_dict)
                contour_plot = hv.Path(xy).opts(style={'color':'white'})
            for metric_name, metric_plot in local_plot_dict.items():
                local_plot = metric_plot.opts(style={'cmap':'Viridis'})
                plot_key = tuple([*ss.dimension_values] + 
                                 [metric_name])
                
                dimension_names = list(self.keys_with_bounds.keys())
                local_plot = local_plot.redim(x=dimension_names[0], y=dimension_names[1])
                if contour_pfaffian is not False:
                    local_plot = local_plot*contour_plot
                if tables is not False:
                    local_plot = local_plot + syst_tables[plot_key] + parameter_tables[plot_key]


                plot_dict[plot_key] = local_plot            
        return (kdims, plot_dict)

    def save(self, folder):
        fname=self.hash_str

        self.learner.save(folder, self.learner_file_prefix + fname)
        
        os.makedirs(folder, exist_ok=True)
        fname = os.path.join(folder, self.aggregate_simulation_set_file_prefix + fname)
        with open(fname, 'wb') as f:
            dill.dump((self.input_params, self.dimension_dict), f)

        
        self.learner.save(folder, fname_pattern=self.learner_file_prefix + self.hash_str + '{}')
        
    def start_periodic_saver(self, runner, folder, interval=3600):
        self.save(folder)
        return self.learner.start_periodic_saver(runner,
                                                 folder,
                                                 fname_pattern=self.learner_file_prefix + self.hash_str + '{}',
                                                 interval=interval)

    def load(self, folder, enough_points):
        self.make_balancing_learner(enough_points)
        self.learner.load(folder, fname_pattern=self.learner_file_prefix + self.hash_str + '{}')

    def load_from_file(fname, folder, enough_points):
        fname = os.path.join(folder, AggregatesSimulationSet.aggregate_simulation_set_file_prefix + fname)

        with open(fname, 'rb') as f:
                (input_params, dimension_dict) = dill.load(f)

        ass = AggregatesSimulationSet(*input_params)
        for k,v in dimension_dict.items():
            ass.add_dimension(k, v)
        ass.load(folder, enough_points)
        return ass