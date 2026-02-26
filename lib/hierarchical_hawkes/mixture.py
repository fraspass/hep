#!/usr/bin/env python3

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans
## Import functions from file hhf.py
from .hhp import nll, fit_process

class HierarchicalHakwesMixture:
    """
    EM algorithm for mixture of Hierarchical Hawkes Point Processes.
    Each sequence of events conditional on latent cluster assignment z_i=k follows a 
    Hierarchical Hawkes Point Process with cluster-specific parameters theta_k.
    Model: Event sequence i | z_i=k ~ HierarchicalHawkes(lambda_k(t; theta_k))    
    Requires: fit_fn (function to fit parameters theta_k from weighted observations)
    """
    def __init__(self, K, fit_fn=None, jumps=None, jumps_alpha=None, 
                    add_delay=False, fixed_jumps=False, fixed_decay=False, fixed_delay=False, 
                    inhibitory_decay=True, inhibitory_delay=True,
                    max_iter=200, tol=1e-6, verbose=True):
        """
        Parameters:
        -----------
        K                 : int, number of mixture components
        fit_fn            : callable, optional function to fit parameters. If None, uses fit_process from hhf.py
        jumps             : array-like, jump times (event times) for the process
        jumps_alpha       : array-like, optional, jump times for alpha parameters (used with delay=True)
        add_delay         : bool, if True uses delayed model with delta parameters
        fixed_jumps       : bool, if True, jump parameters are fixed
        fixed_decay       : bool, if True, decay parameters are fixed
        fixed_delay       : bool, if True, delay parameters are fixed (only used if add_delay=True)
        inhibitory_decay  : bool, if True, decay parameters are inhibitory; otherwise, excitatory
        inhibitory_delay  : bool, if True, delay parameters are inhibitory; otherwise, excitatory
        max_iter          : int, maximum number of EM iterations
        tol               : float, convergence tolerance for log-likelihood
        verbose           : bool, whether to print progress information
        """
        # Check if K is an integer
        if not isinstance(K, int):
            raise ValueError("K must be an integer.")
        self.K = int(K)
        # Validate and store boolean flags
        for param_name, param_val in [("add_delay", add_delay), ("fixed_jumps", fixed_jumps),
                                       ("fixed_decay", fixed_decay), ("fixed_delay", fixed_delay),
                                       ("inhibitory_decay", inhibitory_decay), ("inhibitory_delay", inhibitory_delay)]:
            if not isinstance(param_val, bool):
                raise TypeError(f"{param_name} must be a boolean value.")
        # Check consistency
        if not add_delay and fixed_delay:
            raise ValueError("fixed_delay can only be True if add_delay is True.")
        self.add_delay = add_delay
        self.fixed_jumps = fixed_jumps
        self.fixed_decay = fixed_decay
        self.fixed_delay = fixed_delay
        self.inhibitory_decay = inhibitory_decay
        self.inhibitory_delay = inhibitory_delay
        # Check if jumps is a numpy array or list
        if jumps is not None and not (isinstance(jumps, np.ndarray) or isinstance(jumps, list)):
            raise ValueError("Jumps must be a numpy array or list.")
        self.jumps = jumps
        # If delay is True, jumps_alpha could be provided, otherwise initialise to jumps
        if self.add_delay:
            if jumps_alpha is None:
                self.jumps_alpha = jumps
            else:
                if not (isinstance(jumps_alpha, np.ndarray) or isinstance(jumps_alpha, list)):
                    raise ValueError("jumps_alpha must be a numpy array or list.")
                self.jumps_alpha = jumps_alpha
        else:
            self.jumps_alpha = None
        # Use fit_process from hhf.py if fit_fn is not provided
        if fit_fn is None:
            self.fit_fn = fit_process
        else:
            self.fit_fn = fit_fn
        # Initialize parameters
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.verbose = bool(verbose)
        # Placeholders to be filled at fitting stage
        self.pi = None          # (K,)
        self.theta = None       # List of process parameters
        self.loglik_trace = []
        # Return output if verbose
        if self.verbose:
            print('Model Summary:')
            print(f'  - Number of clusters (K): {self.K}')
            print(f'  - Jump locations: {self.jumps}')
            if self.jumps_alpha is not None:
                print(f'  - Jump locations (alpha): {self.jumps_alpha}')
            print(f'  - Add delay: {self.add_delay}')
            print(f'  - Fixed jumps: {self.fixed_jumps}')
            print(f'  - Fixed decay: {self.fixed_decay}')
            if self.add_delay:
                print(f'  - Fixed delay: {self.fixed_delay}')
            print(f'  - Inhibitory decay: {self.inhibitory_decay}')
            print(f'  - Inhibitory delay: {self.inhibitory_delay}')

    ## Initialise all parameters
    def init_params(self, event_times, observation_intervals=None, init_method='kmeans', offset=1e-12, random_state=None):
        """
        Initialise all parameters: mixture proportions and cluster-specific HHP parameters.

        Parameters
        ----------
        event_times : list of 1D arrays
            Each array contains event times for a sequence.
        observation_intervals : array-like, optional
            Observation intervals for each sequence.
        init_method : str
            'kmeans' or 'random' for initialization method.
        offset : float
            Small constant added for numerical stability.
        random_state : int or None
            Random seed for reproducibility.            
        """
        n = len(event_times)
        K = self.K
        # Check if self.T is set
        if hasattr(self, 'T') and self.T is not None:
            T = self.T
        else:
            T = int(np.ceil(max([max(times) if len(times) > 0 else 0 for times in event_times]))) + 1.0
            self.T = T
        ## Check if self.observation_intervals is set
        if hasattr(self, 'obs_intervals') and observation_intervals is not None:
            raise ValueError("observation_intervals has already been set. To recalculate, please create a new instance of the class.")
        else:
            if observation_intervals is not None:
                ## Check that observation_intervals is an array or list
                observation_intervals = np.asarray(observation_intervals)
                if observation_intervals.ndim != 2 or observation_intervals.shape[1] != 2:
                    raise ValueError("observation_intervals must be a 2D array with shape (R, 2).")
                self.obs_intervals = observation_intervals
                self.total_observed_time = np.sum(observation_intervals[:, 1] - observation_intervals[:, 0])
            else:
                self.obs_intervals = observation_intervals
                self.total_observed_time = T
        # Use k-means on histogram estimates for initial clustering
        if init_method == 'kmeans':
            # Create histogram estimates of intensity for each sequence
            n_bins = min(100, int(T))
            bin_edges = np.linspace(0, T, n_bins + 1)
            intensity_histograms = np.zeros((n, n_bins))
            for i in range(n):
                if len(event_times[i]) > 0:
                    counts, _ = np.histogram(event_times[i], bins=bin_edges)
                    # Normalize to rate (events per unit time)
                    bin_width = bin_edges[1] - bin_edges[0]
                    intensity_histograms[i] = counts / bin_width
            kmeans = KMeans(n_clusters=K, random_state=random_state).fit(intensity_histograms)
            labels = kmeans.labels_
            ## r = np.zeros((n, K))
            ## r[np.arange(n), labels] = 1.0
            r = 0.2 / (K - 1) * np.ones((n, K)) if K > 1 else np.ones((n, 1))
            if K > 1:
                r[np.arange(n), labels] = 0.8
        else:
            if random_state is not None:
                np.random.seed(random_state)
            # Random soft assignments from Dirichlet distribution
            r = np.random.dirichlet(np.ones(K), size=n)
        # Initialize mixture proportions
        self.pi = (r.sum(axis=0) + offset) / n
        # Calculate number of parameters based on configuration
        n_params = 3  # lambda0, theta0_alpha, theta0_beta (base)
        if not self.fixed_jumps:
            n_params += 2  # eta_alpha, xi_alpha
        if not self.fixed_decay:
            n_params += 2  # eta_beta, xi_beta
        if self.add_delay:
            n_params += 1  # theta0_delta
            if not self.fixed_delay:
                n_params += 2  # eta_delta, xi_delta
        # Initialize Hawkes process parameters for each cluster
        self.theta = np.zeros((K, n_params))
        for k in range(K):
            # Weighted counts and lengths accounting for observation intervals
            weighted_counts = 0.0
            weighted_lengths = 0.0
            # Loop over all sequences
            for i in range(n):
                w  = r[i, k]                      # Weight
                if w < 1e-6:                      # Skip if negligible contribution
                    continue
                # Total observed time for this sequence
                N_i = len(event_times[i])   # Number of events
                # Accumulate weighted counts and lengths
                weighted_counts  += w * N_i
                if isinstance(self.total_observed_time, np.ndarray):
                    weighted_lengths += w * self.total_observed_time[i]
                else:
                    weighted_lengths += w * self.total_observed_time
            # Compute weighted mean event rate
            if weighted_lengths > 0:
                r_mean = weighted_counts / weighted_lengths   # weighted mean rate per unit observed time
            else:
                r_mean = 0.1   # fallback safe default if no events or no observed time
            # Initialize parameters based on configuration
            params_list = []
            # Baseline intensity
            params_list.append(np.log(max(1e-8, 0.2 * r_mean)))  # lambda0
            # Jump excitation parameters (alpha)
            params_list.append(np.log(max(1e-8, 0.5 * r_mean)))  # theta0_alpha
            if not self.fixed_jumps:
                params_list.append(np.log(max(1e-12, 0.1 * 0.5 * r_mean)))  # eta_alpha
                params_list.append(np.log(0.5))  # xi_alpha
            # Decay parameters (beta)
            params_list.append(np.log(1.0))  # theta0_beta
            if not self.fixed_decay:
                params_list.append(np.log(0.1))  # eta_beta
                params_list.append(np.log(0.5))  # xi_beta
            # Delay parameters (delta) if applicable
            if self.add_delay:
                params_list.append(np.log(1))  # theta0_delta
                if not self.fixed_delay:
                    params_list.append(np.log(0.005))  # eta_delta
                    params_list.append(np.log(0.5)) # xi_delta
            params = np.array(params_list)
            # Verify parameter count matches expected
            if len(params) != n_params:
                raise ValueError(f"Parameter initialization mismatch: expected {n_params}, got {len(params)}")
            self.theta[k] = params
        # M-step: update all parameters
        S_k = r.sum(axis=0) + offset  # Effective cluster sizes (K,)
        self.pi = S_k / n  # Update mixture proportions
        # Convert list to dictionary for fit_fn if needed
        if isinstance(event_times, list):
            event_times_dict = {i: event_times[i] for i in range(len(event_times))}
        else:
            event_times_dict = event_times
        # Update Hawkes process parameters for each cluster
        for k in range(self.K):
            # Start from previous parameters
            theta_prev = np.copy(self.theta[k])
            # Call fit_fn with correct parameters from hhf.py
            fit_result = self.fit_fn(init=theta_prev, event_times=event_times_dict, jumps=self.jumps, T=self.T, 
                                    weights=r[:, k], observation_intervals=self.obs_intervals, 
                                    sum_values=True, add_delay=self.add_delay, fixed_jumps=self.fixed_jumps,
                                    fixed_decay=self.fixed_decay, fixed_delay=self.fixed_delay,
                                    inhibitory_decay=self.inhibitory_decay, inhibitory_delay=self.inhibitory_delay)
            self.theta[k] = fit_result.x  # Update parameters

    ## Set or calculate the maximum observation time T
    def calculate_T(self, event_times, T=None):
        ## If self.T already exists, raise an error
        if hasattr(self, 'T'):
            raise ValueError("T has already been set. To recalculate T, please create a new instance of the class.")
        if T is None:
            # Determine T from the maximum event time across all sequences
            T = int(np.ceil(max([max(event_times[node]) if len(event_times[node]) > 0 else 0 for node in event_times])))
        else:
            # Check if T is a float or integer, larger than all event times
            max_event = max([max(event_times[node]) if len(event_times[node]) > 0 else 0 for node in event_times])
            if not isinstance(T, (int, float)) or not T > 0 or T <= max_event:
                raise ValueError("T must be a positive number, greater than all event times.")
        self.T = T
        return T

    ## Compute responsibilities (posterior probabilities) for each observation.
    def E_step(self, event_times, observation_intervals=None):
        """
        E-step: Compute responsibilities (posterior probabilities).
        
        Parameters
        ----------
        event_times : list of 1D arrays or dict
            Event times for each sequence.
        observation_intervals : array-like, optional
            Observation intervals.
            
        Returns
        -------
        resp : ndarray, shape (n, K)
            Posterior responsibilities for each sequence and cluster.
        ll : float
            Log-likelihood contribution from this step.
        """
        ## Number of sequences
        n = len(event_times)
        ## Check if self.T is set
        if hasattr(self, 'T') and self.T is not None:
            T = self.T
        else:
            T = int(np.ceil(max([max(times) if len(times) > 0 else 0 for times in event_times]))) + 1.0
            self.T = T
        ## Check if self.observation_intervals is set
        if hasattr(self, 'obs_intervals') and observation_intervals is not None:
            raise ValueError("observation_intervals has already been set. To recalculate, please create a new instance of the class.")
        else:
            if observation_intervals is not None:
                ## Check that observation_intervals is an array or list
                observation_intervals = np.asarray(observation_intervals)
                if observation_intervals.ndim != 2 or observation_intervals.shape[1] != 2:
                    raise ValueError("observation_intervals must be a 2D array with shape (R, 2).")
                self.obs_intervals = observation_intervals
                self.total_observed_time = np.sum(observation_intervals[:, 1] - observation_intervals[:, 0])
            else:
                self.obs_intervals = observation_intervals
                self.total_observed_time = T
        # Prepare data matrix for responsibilities
        K = self.K
        log_resp = np.zeros((n, K))
        # Convert list of arrays to dictionary if needed for nll function
        if isinstance(event_times, list):
            event_times_dict = {i: event_times[i] for i in range(len(event_times))}
        else:
            event_times_dict = event_times
        # Compute log p(y_i | z_i=k) + log pi_k for each cluster k
        for k in range(K):
            # Use the negative log-likelihood function with correct parameters
            nll_vals = nll(params=self.theta[k], event_times=event_times_dict, jumps=self.jumps, T=T, 
                          weights=None, observation_intervals=self.obs_intervals, sum_values=False,
                          add_delay=self.add_delay, fixed_jumps=self.fixed_jumps, 
                          fixed_decay=self.fixed_decay, fixed_delay=self.fixed_delay,
                          inhibitory_decay=self.inhibitory_decay, inhibitory_delay=self.inhibitory_delay)
            log_resp[:, k] = -nll_vals + np.log(self.pi[k] + 1e-12)
        # Normalize probabilities with logsumexp
        lse = logsumexp(log_resp, axis=1, keepdims=True)
        resp = np.exp(log_resp - lse)
        return resp, lse.sum()  # responsibilities and log-likelihood contribution

    def fit(self, event_times, observation_intervals=None, T=None, init_method='kmeans', offset=1e-12, random_state=None):
        """
        Fit the mixture model using the EM algorithm.
        
        Parameters:
        -----------
        event_times : list of arrays, dict, or ndarray
            Event times for each sequence. Can be a list of arrays or a dictionary.
        observation_intervals : array-like, optional
            Observation intervals for each sequence.
        T : float, optional
            Maximum observation time. If None, computed from data.
        init_method : str
            'kmeans' or 'random' for initialization method.
        offset : float
            Small constant added for numerical stability.
        random_state : int or None
            Random seed for reproducibility.
        """
        # Convert event_times to list format if it's a dictionary
        if isinstance(event_times, dict):
            event_times_list = [event_times[i] for i in range(len(event_times))]
        else:
            event_times_list = list(event_times)
        ## Number of sequences
        n = len(event_times_list)
        # Handle T: allow multiple calls to fit with the same or compatible T
        if T is not None:
            if hasattr(self, 'T') and self.T is not None:
                if T != self.T:
                    raise ValueError(f"T value mismatch: fit() already called with T={self.T}, but now called with T={T}.")
                # Same T, allow re-fitting with different random seeds
            else:
                # First call to fit
                self.T = T
        else:
            if hasattr(self, 'T') and self.T is not None:
                # Use previously set T
                T = self.T
            else:
                # Determine T from the maximum event time across all sequences
                T = int(np.ceil(max([max(times) if len(times) > 0 else 0 for times in event_times_list]))) + 1.0
                self.T = T
        ## Check if self.observation_intervals is set
        if hasattr(self, 'obs_intervals') and observation_intervals is not None:
            raise ValueError("observation_intervals has already been set. To recalculate, please create a new instance of the class.")
        else:
            if observation_intervals is not None:
                ## Check that observation_intervals is an array or list
                observation_intervals = np.asarray(observation_intervals)
                if observation_intervals.ndim != 2 or observation_intervals.shape[1] != 2:
                    raise ValueError("observation_intervals must be a 2D array with shape (R, 2).")
                self.obs_intervals = observation_intervals
                self.total_observed_time = np.sum(observation_intervals[:, 1] - observation_intervals[:, 0])
            else:
                self.obs_intervals = observation_intervals
                self.total_observed_time = self.T
        # Initialize all parameters
        self.init_params(event_times_list, observation_intervals=observation_intervals, 
                        init_method=init_method, offset=offset, random_state=random_state)
        # Reset log-likelihood trace for this fit (allows multiple calls to fit)
        self.loglik_trace = []
        # Track previous log-likelihood for convergence checking
        prev_ll = -np.inf
        for it in range(self.max_iter):
            # E-step
            resp, ll = self.E_step(event_times_list, observation_intervals=observation_intervals)
            self.loglik_trace.append(ll)
            # Check convergence
            if self.verbose:
                print(f"Iteration: {it:3d}\t Log-likelihood: {ll:.6f}\t\t\t", end='\r')
            if it > 0 and np.abs(ll - prev_ll) < self.tol * max(1.0, np.abs(prev_ll)):
                if self.verbose:
                    print()
                break
            prev_ll = ll
            # M-step: update all parameters
            S_k = resp.sum(axis=0) + offset  # Effective cluster sizes (K,)
            self.pi = S_k / n  # Update mixture proportions
            # Convert list to dictionary for fit_fn if needed
            if isinstance(event_times_list, list):
                event_times_dict = {i: event_times_list[i] for i in range(len(event_times_list))}
            else:
                event_times_dict = event_times_list
            # Update Hawkes process parameters for each cluster
            for k in range(self.K):
                # Start from previous parameters
                theta_prev = np.copy(self.theta[k])
                # Call fit_fn with correct parameters from hhf.py
                fit_result = self.fit_fn(init=theta_prev, event_times=event_times_dict, jumps=self.jumps, T=self.T, 
                                        weights=resp[:, k], observation_intervals=self.obs_intervals, 
                                        sum_values=True, add_delay=self.add_delay, fixed_jumps=self.fixed_jumps,
                                        fixed_decay=self.fixed_decay, fixed_delay=self.fixed_delay,
                                        inhibitory_decay=self.inhibitory_decay, inhibitory_delay=self.inhibitory_delay)
                self.theta[k] = fit_result.x  # Update parameters
        if self.verbose:
            print()
            print(f"Final log-likelihood: {ll:.6f}")