import distributed_sns
import sns_system

import kwant

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import holoviews as hv

def plot_syst(syst_pars, params, a_new=None, num_lead_cells=4):
    a = syst_pars['a'] if a_new==None else a_new
    Ll = syst_pars['Ll']
    Lm = syst_pars['Lm']
    Lr = syst_pars['Lr']
    Ly = syst_pars['Ly']
    
    syst = sns_system.make_sns_system(a=a, Ll=Ll, Lm=Lm, Lr=Lr, Ly=Ly)
    
    def delta(sites):
        return [np.abs(syst.hamiltonian(i, i, params=params)[1, 0])
                    for i, site in enumerate(sites)]

    return kwant.plot(syst, num_lead_cells=num_lead_cells, site_color=delta(syst.sites),
                 fig_size=(5, 5), show=False)

def plot_pfaffian(params, results, figsize = (5,5)):
    # Plots pfaffian for range of parameters, for a list of parameters
	(iterable_keys, param_list) = distributed_sns.get_list_of_parameter_dictionaries(params)
	dimLength = []
	for k in iterable_keys:
		dimLength.append(len(params[k]))
		
	if len(iterable_keys) == 1:
		plt.figure(figsize = figsize)
		plt.plot(params[iterable_keys[0]], results)
		plt.xlabel(iterable_keys[0])
		plt.show()

	elif len(iterable_keys) == 2:
		M = np.reshape(results, tuple(dimLength)).T
		
		plt.figure(figsize = figsize)
		plt.xlabel(iterable_keys[0])
		plt.ylabel(iterable_keys[1])
		
		if(iterable_keys[1] == 'phase'):
			plt.yticks([0, 2*np.pi], ['$0$', '$2 \pi$'])
		

		im = plt.imshow(M, extent = [params[iterable_keys[0]][0], params[iterable_keys[0]][-1], 
									params[iterable_keys[1]][0], params[iterable_keys[1]][-1]],
						  aspect =  (len(params[iterable_keys[1]]) / len(params[iterable_keys[0]])*
						  			(params[iterable_keys[0]][-1] / params[iterable_keys[1]][-1]))
						  )
		# plt.colorbar(im)
		plt.show()

def print_blearner_status(bl):
    learners = bl.learners
    for learner in bl.learners:
        print('pars:', end="")
        for par in learner.pars:
            print(f'\t{par:.2f}', end="")
        
        print(f'\tnpoints: {learner.npoints}\tloss: {learner.loss():.2f}')

def plot_band_result(momenta, energies, **opts):
	"""
	Plots results of kwant.Bands using holoviews
	"""
	nbands = len(energies[0])
	band_struct = [(momenta, [energies[k][band] for k in range(len(momenta))]) for band in range(nbands)]
	plot_dict = {band : hv.Curve(band_struct[band]) for band in range(nbands)}
	return hv.NdOverlay(plot_dict, **opts)

def get_linked_plots(updatable_plot_function, clickable_plot, clickable_data, default_xy=None, **params):
	if default_xy==None:
		click = hv.streams.Tap(source=clickable_plot)
	else:
		(x, y) = default_xy
		click = hv.streams.Tap(source=clickable_plot, x=x, y=y)

	plot_func = lambda x, y: updatable_plot_function(
		find_closest_key_in_dict((x, y), clickable_data.keys()),
		**params)

	return hv.DynamicMap(plot_func, streams=[click])


def find_closest_key_in_dict(point, xdata):
	""" 
	Finds key closest to give 2-dimensional point in xdata
	"""
	min_err = np.inf
	xmax = -np.inf
	ymax = -np.inf
	xmin = np.inf
	ymin = np.inf

	# Get scaling
	for xy in xdata:
		(x, y) = xy
		xmax = max(xmax, x)
		ymax = max(ymax, y)
		xmin = min(xmin, x)
		ymin = min(ymin, y)
	
	xscale = (xmax - xmin)**2
	yscale = (ymax - ymin)**2
	
	# Get closest point
	for xy in xdata:
		cur_err = (xy[1] - point[1])**2/yscale + (xy[0] - point[0])**2/xscale
		if cur_err < min_err:
			min_err = cur_err
			min_i = xy
	return min_i

