#!/usr/bin/env python3

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from scipy.optimize import minimize
## Import functions from file hhf.py
from .hhf import calculate_intensity, nll

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
                    inhibitory_decay=False, inhibitory_delay=False,
                    max_iter=200, tol=1e-6, verbose=True):
        """
        Parameters:
        -----------
        K              : int, number of mixture components
        fit_fn         : callable, fit_fn(y, w, jumps, init) -> optimization result with .x attribute
        jumps          : array-like, jump times (event times) for the process
        jumps_alpha    : array-like, optional, jump times for alpha parameters (used with delay=True)
        delay          : bool, if True uses 10-parameter delayed model, otherwise 7-parameter model
        max_iter       : int, maximum number of EM iterations
        tol            : float, convergence tolerance for log-likelihood
        verbose        : bool, whether to print progress information
        """
        # Check if K is an integer
        if not isinstance(K, int):
            raise ValueError("K must be an integer.")
        self.K = int(K)
        # Check if delay is a boolean
        if not isinstance(add_delay, bool):
            raise ValueError("Delay must be a boolean (True/False).")
        self.delay = add_delay
        # Check if jumps is a numpy array or list
        if jumps is not None and not (isinstance(jumps, np.ndarray) or isinstance(jumps, list)):
            raise ValueError("Jumps must be a numpy array or list.")
        self.jumps = jumps
        # If delay is True, jumps_alpha could be provided, otherwise initialise to jumps
        if self.delay:
            if jumps_alpha is None:
                self.jumps_alpha = jumps
            else:
                if not (isinstance(jumps_alpha, np.ndarray) or not isinstance(jumps_alpha, list)):
                    raise ValueError("Jumps_alpha must be a numpy array or list.")
                self.jumps_alpha = jumps_alpha
        else:
            self.jumps_alpha = None
        # Check that fit_fn and predict_fn are provided
        if fit_fn is None:
            raise ValueError("The function fit_fn must be provided.")
        # Initialize parameters
        self.fit_fn = fit_fn
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
            print(f'  - Delay in jumps: {self.delay}')

    ## Initialise all parameters
    def init_params(self, event_times, observation_intervals=None, init_method='kmeans', offset=1e-12, random_state=None):
        """
        Initialise all parameters: mixture proportions and cluster-specific HHP parameters.

        Parameters
        ----------
        event_times : list of 1D arrays
            Each array contains event times for a sequence.
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
                if not isinstance(observation_intervals, list) or not isinstance(observation_intervals, np.ndarray) or observation_intervals.shape[1] != 2:
                    raise ValueError("observation_intervals must be a list or numpy array. Each element should be an array of shape 2.")
                else:
                    self.obs_intervals = observation_intervals
                    self.total_observed_time = np.array([np.sum(intervals[:,1] - intervals[:,0]) for intervals in observation_intervals])
            else:
                self.obs_intervals = observation_intervals
                self.total_observed_time = self.T
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
            r = 0.5 / (K - 1) * np.ones((n, K)) if K > 1 else np.ones((n, 1))
            if K > 1:
                r[np.arange(n), labels] = 0.5
        else:
            if random_state is not None:
                np.random.seed(random_state)
            # Random soft assignments from Dirichlet distribution
            r = np.random.dirichlet(np.ones(K), size=n)
        # Initialize mixture proportions
        self.pi = (r.sum(axis=0) + offset) / n
        # Initialize Hawkes process parameters for each cluster
        self.theta = []
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
                weighted_counts  += w * N_i          # weighted number of events
                weighted_lengths += w * self.total_observed_time # weighted total observed time
            # Compute weighted mean event rate
            if weighted_lengths > 0:
                r_mean = weighted_counts / weighted_lengths   # weighted mean rate per unit observed time
            else:
                r_mean = 0.1   # fallback safe default if no events or no observed time
            # Baseline intensity
            lambda0 = max(1e-8, 0.2 * r_mean)
            # Jump excitation parameters (per-changepoint baseline)
            theta0_alpha = max(1e-8, 0.5 * r_mean)
            eta_alpha = 0.1 * theta0_alpha
            xi_alpha = eta_alpha + 0.5
            # Decay parameters
            theta0_beta = 1.0
            eta_beta = 0.1 * theta0_beta
            xi_beta = eta_beta + 0.5
            # Delay parameters if applicable
            if self.delay:
                # 10-parameter model with delta
                theta0_delta = 0.05  # seconds, typical small delay
                eta_delta = 0.1 * theta0_delta
                xi_delta = eta_delta + 0.5
                ## Group parameters
                params = np.log([
                    lambda0,
                    theta0_alpha, eta_alpha, xi_alpha,
                    theta0_beta, eta_beta, xi_beta,
                    theta0_delta, eta_delta, xi_delta
                ])
            else:
                # 7-parameter model
                params = np.log([
                    lambda0,
                    theta0_alpha, eta_alpha, xi_alpha,
                    theta0_beta, eta_beta, xi_beta
                ])
            # Fit initial parameters
            ## theta = self.fit_fn(event_times=event_times, jumps=self.jumps, T=self.T, weights=r[:, k], 
            ##                     observation_intervals=self.obs_intervals, sum_values=False, init=params).x
            ## self.theta.append(theta)
            self.theta.append(params)  # Use initial params directly without fitting

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
                if not isinstance(observation_intervals, list) or not isinstance(observation_intervals, np.ndarray) or observation_intervals.shape[1] != 2:
                    raise ValueError("observation_intervals must be a list or numpy array. Each element should be an array of shape 2.")
                else:
                    self.obs_intervals = observation_intervals
                    self.total_observed_time = np.array([np.sum(intervals[:,1] - intervals[:,0]) for intervals in observation_intervals])
            else:
                self.obs_intervals = observation_intervals
                self.total_observed_time = self.T
        # Prepare data matrix for responsibilities
        K = self.K
        log_resp = np.zeros((n, K))
        # Compute log p(y_i | z_i=k) + log pi_k for each cluster k
        for k in range(K):
            # Use the negative log-likelihood function
            log_resp_k = nll(params=self.theta[k], event_times=event_times, jumps=self.jumps, T=T, weights=None, 
                             observation_intervals=self.obs_intervals, sum_values=False)  
            log_resp[:, k] = -log_resp_k + np.log(self.pi[k] + 1e-12)  # Add small constant for numerical stability
        # Normalize probabilities with logsumexp
        lse = logsumexp(log_resp, axis=1, keepdims=True)
        resp = np.exp(log_resp - lse)
        return resp, lse.sum()  # responsibilities and log-likelihood contribution

    def fit(self, event_times, observation_intervals=None, T=None, init_method='kmeans', offset=1e-12, random_state=None):
        """
        Fit the mixture model using the EM algorithm.
        
        Parameters:
        -----------
        event_times : dictionary of lists or np.ndarray, each list contains event times for a sequence
        init_method : str, 'kmeans' or 'random' for initialization method
        offset      : float, small constant added for numerical stability
        random_state: int or None, random seed for reproducibility
        """
        n = len(event_times)
        if hasattr(self, 'T') and T is not None:
            raise ValueError("T has already been set. To recalculate T, please create a new instance of the class.")
        else:
            if T is None:
                # Determine T from the maximum event time across all sequences
                T = int(np.ceil(max([max(event_times[node]) if len(event_times[node]) > 0 else 0 for node in event_times])))
                self.T = T
            else:
                # Check if T is a float or integer, larger than all event times
                max_event = max([max(event_times[node]) if len(event_times[node]) > 0 else 0 for node in event_times])
                if not isinstance(T, (int, float)) or not T > 0 or T <= max_event:
                    raise ValueError("T must be a positive number, greater than all event times.")
                self.T = T
        ## Check if self.observation_intervals is set
        if hasattr(self, 'obs_intervals') and observation_intervals is not None:
            raise ValueError("observation_intervals has already been set. To recalculate, please create a new instance of the class.")
        else:
            if observation_intervals is not None:
                ## Check that observation_intervals is an array or list
                if not isinstance(observation_intervals, list) or not isinstance(observation_intervals, np.ndarray) or observation_intervals.shape[1] != 2:
                    raise ValueError("observation_intervals must be a list or numpy array. Each element should be an array of shape 2.")
                else:
                    self.obs_intervals = observation_intervals
                    self.total_observed_time = np.array([np.sum(intervals[:,1] - intervals[:,0]) for intervals in observation_intervals])
            else:
                self.obs_intervals = observation_intervals
                self.total_observed_time = self.T
        # Initialize all parameters
        resp = self.init_params(event_times, init_method=init_method, offset=offset, random_state=random_state)
        # Track previous log-likelihood for convergence checking
        prev_ll = -np.inf
        for it in range(self.max_iter):
            # E-step
            resp, ll = self.E_step(event_times)
            self.loglik_trace.append(ll)
            # Check convergence
            if self.verbose:
                print(f"Iteration: {it:3d}\t Log-likelihood: {ll:.6f}\t\t\t", end='\r')
            if it > 0 and np.abs(ll - prev_ll) < self.tol * max(1.0, np.abs(prev_ll)):
                break
            prev_ll = ll
            # M-step: update all parameters
            S_k = resp.sum(axis=0) + offset  # Effective cluster sizes (K,)
            self.pi = S_k / n  # Update mixture proportions
            # Update Hawkes process parameters for each cluster
            for k in range(self.K):
                # Compute observation weights (inverse variance weighted by responsibilities)(n, T)
                theta_prev = np.copy(self.theta[k])
                # Warm start from previous parameters
                fit = self.fit_fn(y=event_times, w=resp[:, k], jumps=self.jumps, init=theta_prev)
                self.theta[k] = fit.x # Update parameters
        if self.verbose:
            print()
            print(f"Final log-likelihood: {ll:.6f}")