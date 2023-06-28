from typing import Optional
import numpy as np


from tick.hawkes import SimuHawkesSumExpKernels, SimuHawkesMulti, SimuHawkesExpKernels, \
    HawkesSumExpKern

    
def initialize_kernel(kernel_name:str,
                       marks:int, 
                       baselines:list, 
                       window:float,
                       decays:Optional[list]=None, 
                       self_decays:Optional[list]=None, 
                       mutual_decays:Optional[list]=None, 
                       adjacency:Optional[list]=None, 
                       self_adjacency:Optional[list]=None, 
                       mutual_adjacency:Optional[list]=None, 
                       noise:Optional[float]=None):
    artifacts = {}
    if kernel_name in ['hawkes_exponential_mutual', 'hawkes_exponential_independent']:
        if decays is None:
            decays = [[self_decays[i] if i == j else mutual_decays for i in range(marks)] for j in range(marks)]
        if adjacency is None:
            adjacency = [[self_adjacency[i] if i == j else mutual_adjacency for i in range(marks)] for j in range(marks)]
        kernel = SimuHawkesExpKernels(adjacency=adjacency, decays=decays, baseline=baselines, end_time=window, verbose=False)
    elif kernel_name in ['hawkes_sum_exponential_mutual', 'hawkes_sum_exponential_independent']:
        if decays is None:
            decays = np.array(self_decays)
        if adjacency is None:
            adjacency = np.array([[self_adjacency[i] if i == j else mutual_adjacency for i in range(marks)] for j in range(marks)])
            adjacency_mask = adjacency != 0
            adjacency = np.repeat(adjacency[:, :, np.newaxis], decays.shape[0] , axis=2)
            adjacency_mask = np.repeat(adjacency_mask[:, :, np.newaxis], decays.shape[0] , axis=2)
            if noise is not None:
                adj_noise = np.random.uniform(0, noise, adjacency.shape) * adjacency_mask
                adjacency = adjacency + adj_noise 
        kernel = SimuHawkesSumExpKernels(adjacency=adjacency, decays=decays, baseline=baselines, end_time=window, verbose=False)
    if kernel.spectral_radius() >= 1:
        kernel.adjust_spectral_radius(0.99)
        print('Spectral radius adjusted !')
    artifacts['decays'] = kernel.decays.tolist()
    artifacts['adjacency'] = kernel.adjacency.tolist()
    return kernel, artifacts

def simulate_process(kernel_name:str, 
                     window:float, 
                     n_seq:int, 
                     marks:int, 
                     baselines:list, 
                     self_decays:Optional[list]=None, 
                     mutual_decays:Optional[list]=None, 
                     self_adjacency:Optional[list]=None, 
                     mutual_adjacency:Optional[list]=None, 
                     track_intensity:Optional[bool]=False, 
                     decays:Optional[list]=None, 
                     adjacency:Optional[list]=None, 
                     noise:Optional[float]=None):
    if decays is None:
        assert(self_decays is not None and mutual_decays is not None), 'Either specify self decays and mutual decays, or overall decays, but not both.'
    if adjacency is None:
        assert(self_adjacency is not None and mutual_adjacency is not None), 'Either specify self adjacency and mutual adjacency, or overall adjacency, but not both.'
    kernel, artifacts = initialize_kernel(kernel_name=kernel_name, marks=marks, self_decays=self_decays, mutual_decays=mutual_decays, 
                                          self_adjacency=self_adjacency, mutual_adjacency=mutual_adjacency, baselines=baselines, window=window,
                                          decays=decays, adjacency=adjacency, noise=noise)
    if track_intensity:
        dt = 0.01
        kernel.track_intensity(dt)
    process = SimuHawkesMulti(kernel, n_simulations=n_seq)
    process.end_time = [window] * n_seq  
    process.simulate()
    return process, artifacts

def get_kernel(name):
    if name == 'hawkes_exponential':
        kernel = SimuHawkesExpKernels
    elif name == 'hawkes_sum_exponential':
        kernel = SimuHawkesSumExpKernels
    return kernel

