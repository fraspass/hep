#!/usr/bin/env python3

## Utility functions for hierarchical Hawkes functions
import numpy as np
from scipy.optimize import minimize

## Inhibitory Hawkes intensity function for the model hyperparameters
def inhibitory_hawkes_intensity(times, jumps, theta0, eta, xi, inhibitory=True):
    ## Check parameter values
    if not (theta0 >= 0 and eta >= 0 and xi >= 0):
        raise ValueError("Parameters theta0, eta, and xi must be non-negative.")
    ## Check input types and lengths
    if not isinstance(times, (list, np.ndarray)) or not isinstance(jumps, (list, np.ndarray)):
        raise TypeError("Times and jumps must be lists or numpy arrays.")
    if len(times) == 0 or len(jumps) == 0:
        raise ValueError("Times and jumps must not be empty.")
    ## Convert inputs to numpy arrays
    times = np.array(times)
    jumps = np.array(jumps)
    if times.ndim != 1 or jumps.ndim != 1:
        raise ValueError("Times and jumps must be one-dimensional arrays.")
    ## Initialize intensity array
    intensity = np.full_like(times, fill_value=theta0, dtype=np.float64)
    ## Update the intensity at each jump 
    for jump in jumps:
        subset_times = (times > jump)
        if inhibitory:
            intensity[subset_times] -= eta * np.exp(-xi * (times[subset_times] - jump))
        else:
            intensity[subset_times] += eta * np.exp(-xi * (times[subset_times] - jump))
    ## Ensure intensity does not drop below zero
    intensity = np.maximum(intensity, 0)
    ## Return the computed intensity
    return intensity

## Construct fit_fn
def fit_process(init, event_times, jumps, T, weights=None, observation_intervals=None, sum_values=False,
                add_delay=False, fixed_jumps=False, fixed_decay=False, fixed_delay=False, inhibitory_decay=True, inhibitory_delay=True):
    # Minimize the loss function
    result = minimize(nll, init, args=(event_times, jumps, T, weights, observation_intervals, sum_values,
                                       add_delay, fixed_jumps, fixed_decay, fixed_delay, inhibitory_decay, inhibitory_delay), method='L-BFGS-B')
    return result

## Function to calculate the intensity given parameters, jumps, evaluation points
def calculate_intensity(params, t, jumps, add_delay=False, fixed_jumps=False, fixed_decay=False, fixed_delay=False, inhibitory_decay=True, inhibitory_delay=True, log_params=False):
    ## Check add_delay, fixed_jumps, fixed_decay, fixed_delay as booleans
    if not isinstance(add_delay, bool):
        raise TypeError("add_delay must be a boolean value.")
    if not isinstance(fixed_jumps, bool):
        raise TypeError("fixed_jumps must be a boolean value.")
    if not isinstance(fixed_decay, bool):
        raise TypeError("fixed_decay must be a boolean value.")
    if not isinstance(fixed_delay, bool):
        raise TypeError("fixed_delay must be a boolean value.")
    if not isinstance(inhibitory_decay, bool):
        raise TypeError("inhibitory_decay must be a boolean value.")
    if not isinstance(inhibitory_delay, bool):
        raise TypeError("inhibitory_delay must be a boolean value.")
    if not isinstance(log_params, bool):
        raise TypeError("log_params must be a boolean value.")
    if not add_delay and fixed_delay:
        raise ValueError("fixed_delay can only be True if add_delay is True.")
    ## Calculate number of model parameters according to setup
    n_params = 3
    if not fixed_jumps:
        n_params += 2
    if not fixed_decay:
        n_params += 2
    if add_delay:
        n_params += 1
        if not fixed_delay:
            n_params += 2
    # Check length of the parameter vector      
    if len(params) != n_params:
        raise ValueError("Expected {} parameters, got {}".format(n_params, len(params)))
    ## Model parameters are only obtained after exponentiation
    params = np.exp(np.clip(params, -100, 100)) if log_params else params
    jumps = np.array(jumps)
    ## Unpack parameters
    lambda0 = params[0]
    theta0_alpha = params[1]
    if not fixed_jumps:
        eta_alpha = params[2]
        csi_alpha = params[3]
        theta0_beta = params[4]
        if not fixed_decay:
            eta_beta = params[5]
            csi_beta = params[6]
            if add_delay:
                theta0_delta = params[7]
                if not fixed_delay:
                    eta_delta = params[8]
                    csi_delta = params[9]
        else:
            if add_delay:
                theta0_delta = params[5]
                if not fixed_delay:
                    eta_delta = params[6]
                    csi_delta = params[7]
    else:
        theta0_beta = params[2]
        if not fixed_decay:
            eta_beta = params[3]
            csi_beta = params[4]
            if add_delay:
                theta0_delta = params[5]
                if not fixed_delay:
                    eta_delta = params[6]
                    csi_delta = params[7]
        else:
            if add_delay:
                theta0_delta = params[3]
                if not fixed_delay:
                    eta_delta = params[4]
                    csi_delta = params[5]
    ## Calculate the model parameters hierarchically
    if fixed_jumps:
        alphas = theta0_alpha * np.ones(len(jumps))
    else:
        alphas = inhibitory_hawkes_intensity(jumps, jumps, theta0_alpha, eta_alpha, csi_alpha)
    if fixed_decay:
        betas = theta0_beta * np.ones(len(jumps))
    else:
        betas = inhibitory_hawkes_intensity(jumps, jumps, theta0_beta, eta_beta, csi_beta, inhibitory=inhibitory_decay)
    if add_delay: 
        if not fixed_delay:
            deltas = inhibitory_hawkes_intensity(jumps, jumps, theta0_delta, eta_delta, csi_delta, inhibitory=inhibitory_delay)
        else:
            deltas = theta0_delta * np.ones(len(jumps))
    ## Calculate the intensity function
    ints = np.ones_like(t) * lambda0
    if add_delay:
        for j, jump in enumerate(jumps):
            c1 = alphas[j] / (deltas[j] + 1e-12) * np.clip(t - jump, 0, deltas[j]) * (1 - np.heaviside(t - jump - deltas[j], 0))
            c2 = np.heaviside(t - jump - deltas[j], 0) * alphas[j] * np.exp(np.clip(-betas[j] * (t - jump - deltas[j]), -100, 100))
            ints += (c1 + c2) * np.heaviside(t - jump, 0) ## Returns 1 if time_grid >= jump, else 0
    else:
        for j, jump in enumerate(jumps):
            ints += alphas[j] * np.exp(np.clip(-betas[j] * (t - jump), -100, 100)) * np.heaviside(t - jump, 0) ## Returns 1 if time_grid >= jump, else 0
    ## Return the intensity function
    return ints

## Maximum likelihood estimation of the parameters of the entire process
def nll(params, event_times, jumps, T, weights=None, observation_intervals=None, sum_values=False, 
        add_delay=False, fixed_jumps=False, fixed_decay=False, fixed_delay=False, inhibitory_decay=True, inhibitory_delay=True):
    ## Check parameter values
    if not isinstance(params, (list, np.ndarray)):
        raise TypeError("Parameters must be a list or numpy array.")
    if not isinstance(event_times, (dict, list, np.ndarray)):
        raise TypeError("Time series must be a list, a numpy array, or a dictionary with n keys.")
    if not isinstance(jumps, (list, np.ndarray)):
        raise TypeError("Jumps must be a list or numpy array.")
    if weights is None:
        if isinstance(event_times, dict):
            weights = np.ones(len(event_times))
    else:
        if not isinstance(event_times, dict):
            raise ValueError("Weights cannot be provided if event_times is not a dictionary.")
        else:
            if not isinstance(weights, (list, np.ndarray)):
                raise TypeError("Weights must be a list or numpy array.")
            if len(weights) != len(event_times):
                raise ValueError("Weights must be of the same length as event_times.")
            weights = np.array(weights)
    ## Check sum_values as boolean
    if not isinstance(sum_values, bool):
        raise TypeError("sum_values must be a boolean value.")
    if sum_values and not isinstance(event_times, dict):
        raise ValueError("sum_values can only be True if event_times is a dictionary.")
    ## Check all boolean flags
    if not isinstance(add_delay, bool):
        raise TypeError("add_delay must be a boolean value.")
    if not isinstance(fixed_jumps, bool):
        raise TypeError("fixed_jumps must be a boolean value.")
    if not isinstance(fixed_decay, bool):
        raise TypeError("fixed_decay must be a boolean value.")
    if not isinstance(fixed_delay, bool):
        raise TypeError("fixed_delay must be a boolean value.")
    if not isinstance(inhibitory_decay, bool):
        raise TypeError("inhibitory_decay must be a boolean value.")
    if not isinstance(inhibitory_delay, bool):
        raise TypeError("inhibitory_delay must be a boolean value.")
    if not add_delay and fixed_delay:
        raise ValueError("fixed_delay can only be True if add_delay is True.")
    ## Check length of the parameter vector according to setup
    n_params = 3
    if not fixed_jumps:
        n_params += 2
    if not fixed_decay:
        n_params += 2
    if add_delay:
        n_params += 1
        if not fixed_delay:
            n_params += 2
    if len(params) != n_params:
        raise ValueError("Expected {} parameters, got {}".format(n_params, len(params)))
    ## Check observation_intervals
    if observation_intervals is not None:
        if not isinstance(observation_intervals, (list, np.ndarray)):
            raise TypeError("Observation intervals must be a list or numpy array.")
        observation_intervals = np.array(observation_intervals)
        if observation_intervals.ndim != 2 or observation_intervals.shape[1] != 2:
            raise ValueError("Observation intervals must be a 2D array with shape (R, 2).")
    ## Convert inputs to numpy arrays to avoid type conflicts
    params = np.exp(np.clip(params, -100, 100))
    n = len(event_times)
    jumps = np.array(jumps)
    ## Unpack parameters
    lambda0 = params[0]
    theta0_alpha = params[1]
    if not fixed_jumps: 
        eta_alpha = params[2]
        csi_alpha = params[3]
        theta0_beta = params[4]
        if not fixed_decay:
            eta_beta = params[5]
            csi_beta = params[6]
            if add_delay:
                theta0_delta = params[7]
                if not fixed_delay:
                    eta_delta = params[8]
                    csi_delta = params[9]
        else:
            if add_delay:
                theta0_delta = params[5]
                if not fixed_delay:
                    eta_delta = params[6]
                    csi_delta = params[7]
    else:
        theta0_beta = params[2]
        if not fixed_decay:
            eta_beta = params[3]
            csi_beta = params[4]
            if add_delay:
                theta0_delta = params[5]
                if not fixed_delay:
                    eta_delta = params[6]
                    csi_delta = params[7]
        else:
            if add_delay:
                theta0_delta = params[3]
                if not fixed_delay:
                    eta_delta = params[4]
                    csi_delta = params[5]
    ## Obtain intensities for event times
    if isinstance(event_times, dict):
        lik1 = {}
        for i in range(n):
            lik1[i] = calculate_intensity(params=params, t=event_times[i], jumps=jumps, add_delay=add_delay, 
                                          fixed_jumps=fixed_jumps, fixed_decay=fixed_decay, fixed_delay=fixed_delay, 
                                          inhibitory_decay=inhibitory_decay, inhibitory_delay=inhibitory_delay, log_params=False)
    else:
        lik1 = calculate_intensity(params=params, t=event_times, jumps=jumps, add_delay=add_delay, 
                                  fixed_jumps=fixed_jumps, fixed_decay=fixed_decay, fixed_delay=fixed_delay, 
                                  inhibitory_decay=inhibitory_decay, inhibitory_delay=inhibitory_delay, log_params=False)
    ## Obtain parameters
    if fixed_jumps:
        alphas = theta0_alpha * np.ones(len(jumps))
    else:
        alphas = inhibitory_hawkes_intensity(jumps, jumps, theta0_alpha, eta_alpha, csi_alpha)
    if fixed_decay:
        betas = theta0_beta * np.ones(len(jumps))
    else:
        betas = inhibitory_hawkes_intensity(jumps, jumps, theta0_beta, eta_beta, csi_beta, inhibitory=inhibitory_decay)
    if add_delay:
        if fixed_delay:
            deltas = theta0_delta * np.ones(len(jumps))
        else:
            deltas = inhibitory_hawkes_intensity(jumps, jumps, theta0_delta, eta_delta, csi_delta, inhibitory=inhibitory_delay)
    ## Calculate integral of the intensity function over [0, T]
    if observation_intervals is None:
        integral = lambda0 * T
        ## Calculate integral of intensity function
        if not add_delay:
            ## Calculate integral for each jump
            for j, jump in enumerate(jumps):
                # I_j = alpha(tau_j) * beta(tau_j)^{-1} * [1 - exp(-beta(tau_j) * (T - tau_j))]
                if betas[j] < 1e-10:
                    integral += alphas[j] * (T - jump)
                else:
                    integral += alphas[j] / betas[j] * (1 - np.exp(-betas[j] * (T - jump)))
        else:
            ## Calculate integral for each jump
            for j, jump in enumerate(jumps):
                if T > jump:
                    time_diff = T - jump
                    if time_diff <= deltas[j]:
                        # I_j = alpha(tau_j) / (2 * delta(tau_j)) * (T - tau_j)^2
                        integral += alphas[j] / (2 * deltas[j]) * (time_diff ** 2)
                    else:
                        # I_j = alpha(tau_j) * delta(tau_j) / 2 + alpha(tau_j) / beta(tau_j) * [1 - exp(-beta(tau_j) * (T - tau_j - delta(tau_j)))]
                        if betas[j] < 1e-10:
                            integral += alphas[j] * deltas[j] / 2 + alphas[j] * (time_diff - deltas[j])
                        else:
                            integral += alphas[j] * deltas[j] / 2 + alphas[j] / betas[j] * (1 - np.exp(-betas[j] * (time_diff - deltas[j])))
    else:
        # Adjust the total integral: instead of [0, T], we only integrate over observation intervals
        integral = 0.0
        for r in range(len(observation_intervals)):
            a_r, b_r = observation_intervals[r]
            # Calculate integral over [a_r, b_r] for the baseline
            integral += lambda0 * (b_r - a_r)
            # Calculate integral for each jump's contribution
            if not add_delay:
                for j, jump in enumerate(jumps):
                    if b_r > jump:
                        # Contribution to [a_r, b_r]
                        # Upper and lower limit contribution
                        if betas[j] < 1e-10:
                            upper = alphas[j] * (b_r - jump) if b_r > jump else 0.0
                            lower = alphas[j] * (a_r - jump) if a_r > jump else 0.0
                        else:
                            upper = alphas[j] / betas[j] * (1 - np.exp(-betas[j] * (b_r - jump))) if b_r > jump else 0.0
                            lower = alphas[j] / betas[j] * (1 - np.exp(-betas[j] * (a_r - jump))) if a_r > jump else 0.0
                        integral += upper - lower
            else:
                for j, jump in enumerate(jumps):
                    if b_r > jump:
                        # Contribution to [a_r, b_r] - upper limit
                        if b_r <= jump:
                            upper = 0.0
                        else:
                            time_diff_upper = b_r - jump
                            if time_diff_upper <= deltas[j]:
                                upper = alphas[j] / (2 * deltas[j]) * (time_diff_upper ** 2)
                            else:
                                upper = alphas[j] * deltas[j] / 2 + alphas[j] / (betas[j]) * (1 - np.exp(-betas[j] * (time_diff_upper - deltas[j])))
                        # Contribution to [a_r, b_r] - lower limit
                        if a_r <= jump:
                            lower = 0.0
                        else:
                            time_diff_lower = a_r - jump
                            if time_diff_lower <= deltas[j]:
                                lower = alphas[j] / (2 * deltas[j]) * (time_diff_lower ** 2)
                            else:
                                lower = alphas[j] * deltas[j] / 2 + alphas[j] / (betas[j]) * (1 - np.exp(-betas[j] * (time_diff_lower - deltas[j])))                        
                        integral += upper - lower
    ## Calculate negative log-likelihood
    if isinstance(event_times, dict):
        # For multiple sequences (mixture model)
        nll_val = 0.0 if sum_values else np.zeros(n)
        for i in range(n):
            # Sum of log intensities at event times
            log_sum = np.sum(np.log(lik1[i] + 1e-12))
            # Weighted negative log-likelihood
            if sum_values:
                nll_val -= weights[i] * (log_sum - integral)
            else:
                nll_val[i] -= weights[i] * (log_sum - integral)
    else:
        # For single sequence
        log_sum = np.sum(np.log(lik1 + 1e-12))
        nll_val = -log_sum + integral
    # Return the (weighted) negative log-likelihood value
    return nll_val