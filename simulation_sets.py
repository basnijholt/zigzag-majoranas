from copy import deepcopy
import collections
from functools import partial
from itertools import product
import numpy as np
import adaptive
import dependencies.adaptive_tools as adaptive_tools

def aggregate_function(xy, point_function, parameters, setup_function, metric_functions):
    setup_data = setup_function(point, parameters, metric_functions)

    metric_results = np.zeros(len(metric_functions))

    parameters_at_point = point(parameters)
    
    for idx, metric_function in enumerate(metric_functions):
        metric_results[idx] = metric_function(setup_data, parameters)
        
    return metric_results

class SimulationSet():
    learner_file_prefix = 'learner_data_'
    simulation_set_file_prefix = 'simulation_set_data_'

    def __init__(self,
                 continuous_parameter_functions,
                 default_parameters,
                 setup_function,
                 metric_functions_dict
                 ):
    """
    input:
        continuous_parameter_functions
            Tuple with parameter names, 2D function altering the parameters, and bounds for the two dimensions.
        default_parameters
            Dictionary containing default parameters
        setup_function
            Function which outputs data to be used for all metrics
        metric_functions_dict
            Dictionary containing functions to be run using parameters and result from setup function

    """

        self.continuous_parameter_functions = deepcopy(continuous_parameter_functions)
        self.default_parameters = deepcopy(default_parameters)
        self.metric_functions_dict = deepcopy(metric_functions_dict)     
        self.setup_function = setup_function

    def get_total_function(self):
        return partial(aggregate_function,
                       point_function=point_function,
                       parameters=parameters,
                       setup_function=setup_function,
                       metric_functions=metric_functions)


    def make_learner(self, loss_function=adaptive.Learner2D.default_learner):
        f = self.get_total_function()
        self.learner = adaptive_tools.Learner2D(f, bounds=self.bounds, loss_per_triangle=loss_function)

    def get_learner(self):
        return self.learner

    @property
    def bounds(self):
        return self.continuous_parameter_functions[2]
    
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

    def get_data_dictionary(self, n):
        ip = self.learner.ip()
        normalized_dim = np.linspace(-.5,.5, n)
        xdim, ydim = self.unnormalize(normalized_dim, normalized_dim)

        gridx, gridy = np.meshgrid(normalized_dim, normalized_dim)
        gridded_results = ip(gridx, gridy)

        data_dictionary = dict()
        for idx, metric_key in enumerate(self.metric_functions):
            metric_results = gridded_results[:,:, idx]
            data_dictionary[metric_key] = (xdim, ydim, metric_results)

        return data_dictionary


class AggregatesSimulationSet():
    learner_file_prefix = 'learner_data_'
    aggregate_simulation_set_file_prefix = 'aggregate_simulation_set_data_'

    def __init__(self,
                 continuous_parameter_functions,
                 default_parameters,
                 setup_function,
                 metric_functions_dict
                 ):
    """
    input:
        continuous_parameter_functions
            Tuple with parameter names, 2D function altering the parameters, and bounds for the two dimensions.
        default_parameters
            Dictionary containing default parameters
        setup_function
            Function which outputs data to be used for all metrics
        metric_functions_dict
            Dictionary containing functions to be run using parameters and result from setup function

    """

        self.dimension_dict = {}

        self.continuous_parameter_functions = deepcopy(continuous_parameter_functions)
        self.default_parameters = deepcopy(default_parameters)
        self.metric_functions_dict = deepcopy(metric_functions_dict)
        self.setup_function = setup_function



    @property
    def learners(self):
        return self.learner.learners

    def add_dimension(self, dimension_name, dimension_functions):
        self.dimension_dict[dimension_name] = dimension_functions

    def make_simulation_sets(self):
        simulation_set_list = []

        for parameter_altering_functions in product(*self.dimension_dict.values()):
            local_parameters = deepcopy(self.default_parameters)
            dimension_values = []

            for function in parameter_altering_functions:
                dimension_values.append(function(local_parameters))

            ss = SimulationSet(self.continuous_parameter_functions,
                               self.default_parameters
                               self.metric_functions_dict,
                               self.setup_function)

            ss.dimension_values = dimension_values
            simulation_set_list.append(ss)

        self.simulation_set_list = simulation_set_list

    def make_learners(self):
        self.make_simulation_sets()
        for ss in self.simulation_set_list:
            ss.make_learner()
            
        learners = [ss.get_learner() for ss in self.simulation_set_list]
        return learners

    def make_balancing_learner(self):
        learners = self.make_learners(enough_points)
        self.learner = adaptive_tools.BalancingLearner(learners)
    
    def get_balancing_learner(self):
        return self.learner

    def get_data_dictionary(self, n):
        data_dictionary = dict()
        kdims = list(self.dimension_dict.keys()) + ['Metric']

        for ss in self.simulation_set_list:
            local_data_dict = ss.get_data_dictionary(n)
            
            for metric_name, metric_data in local_data_dict.items():
                data_key = tuple([*ss.dimension_values] + 
                                 [metric_name])

                data_dictionary[data_key] = metric_data            
        return (kdims, data_dictionary)