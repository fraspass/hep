#!/usr/bin/env python3

## Utility functions for hierarchical Hawkes functions
import numpy as np
from scipy.optimize import minimize

## Inhibitory Hawkes intensity function for the model hyperparameters
def inhibitory_hawkes_intensity(times, jumps, theta0, eta, xi):
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
        intensity[subset_times] -= eta * np.exp(-xi * (times[subset_times] - jump))
    ## Ensure intensity does not drop below zero
    intensity = np.maximum(intensity, 0)
    ## Return the computed intensity
    return intensity

## Construct fit_fn and predict_fn to pass to TimeSeriesMixtureEM in mixture.py
def fit_process(y, w, jumps, init):
    # Minimize the loss function
    result = minimize(nll, init, args=(y, jumps, w), method='L-BFGS-B')
    return result

## Function to calculate the intensity given parameters, jumps, evaluation points
def calculate_intensity(params, t, jumps):
    # Check length of the parameter vector
    if len(params) != 7 and len(params) != 10:
        raise ValueError("Expected 7 or 10 parameters, got {}".format(len(params)))
    ## Model parameters are only obtained after exponentiation
    params = np.clip(params, -100, 100)
    if len(params) == 7:
        lambda0, theta0_alpha, eta_alpha, csi_alpha, theta0_beta, eta_beta, csi_beta = np.exp(params)
    else:
        lambda0, theta0_alpha, eta_alpha, csi_alpha, theta0_beta, eta_beta, csi_beta, theta0_delta, eta_delta, csi_delta = np.exp(params)
    ## Calculate the model parameters hierarchically
    alphas = inhibitory_hawkes_intensity(jumps, jumps, theta0_alpha, eta_alpha, csi_alpha)
    betas = inhibitory_hawkes_intensity(jumps, jumps, theta0_beta, eta_beta, csi_beta)
    if len(params) == 10:
        deltas = inhibitory_hawkes_intensity(jumps, jumps, theta0_delta, eta_delta, csi_delta)
    ## Calculate the intensity function
    ints = np.ones_like(t) * lambda0
    if len(params) == 10:
        for j, jump in enumerate(jumps):
            c1 = alphas[j] / (deltas[j] + 1e-12) * np.clip(t - jump, 0, deltas[j]) * (1 - np.heaviside(t - jump - deltas[j], 1))
            c2 = np.heaviside(t - jump - deltas[j], 1) * alphas[j] * np.exp(np.clip(-betas[j] * (t - jump - deltas[j]), -100, 100))
            ints += (c1 + c2) * np.heaviside(t - jump, 1) ## Returns 1 if time_grid >= jump, else 0
    else:
        for j, jump in enumerate(jumps):
            ints += alphas[j] * np.exp(np.clip(-betas[j] * (t - jump), -100, 100)) * np.heaviside(t - jump, 1) ## Returns 1 if time_grid >= jump, else 0
    ## Return the intensity function
    return ints

## Maximum likelihood estimation of the parameters of the entire process
def nll(params, event_times, jumps, T, weights=None, observation_intervals=None, sum_values=False):
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
        if isinstance(weights, (list, np.ndarray)):
            raise ValueError("Weights cannot be provided if event_times is not a dictionary.")
        else:    
            if len(weights) != len(event_times):
                raise ValueError("Weights must be of the same length as event_times.")
    ## Check sum_values as boolean
    if not isinstance(sum_values, bool):
        raise TypeError("sum_values must be a boolean value.")
    if sum_values and not isinstance(event_times, dict):
        raise ValueError("sum_values can only be True if event_times is a dictionary.")
    ## Check length of the parameter vector
    if len(params) != 7 and len(params) != 10:
        raise ValueError("Expected 7 or 10 parameters, got {}".format(len(params)))
    if observation_intervals is not None:
        if not isinstance(observation_intervals, (list, np.ndarray)):
            raise TypeError("Observation intervals must be a list or numpy array.")
        observation_intervals = np.array(observation_intervals)
        if observation_intervals.ndim != 2 or observation_intervals.shape[1] != 2:
            raise ValueError("Observation intervals must be a 2D array with shape (R, 2).")
    ## Convert inputs to numpy arrays to avoid type conflicts
    params = np.clip(params, -100, 100)
    n = len(event_times)
    jumps = np.array(jumps)
    ## Obtain intensities for event times
    if isinstance(event_times, dict):
        lik1 = {}
        for i in range(n):
            lik1[i] = calculate_intensity(params, event_times[i], jumps)
    else:
        lik1 = calculate_intensity(params, event_times, jumps)
    ## Calculate integral of intensity function
    if len(params) == 7:
        ## Unpack parameters for 7-parameter model (scaled exponential)
        lambda0, theta0_alpha, eta_alpha, csi_alpha, theta0_beta, eta_beta, csi_beta = np.exp(params)
        ## Calculate hierarchical parameters
        alphas = inhibitory_hawkes_intensity(jumps, jumps, theta0_alpha, eta_alpha, csi_alpha)
        betas = inhibitory_hawkes_intensity(jumps, jumps, theta0_beta, eta_beta, csi_beta)
        ## Calculate integral for each jump (Equation 2 from the paper)
        integral = 0.0
        for j, jump in enumerate(jumps):
            if T > jump:
                # I_j = alpha(tau_j) * beta(tau_j)^{-1} * [1 - exp(-beta(tau_j) * (T - tau_j))]
                integral += alphas[j] / (betas[j] + 1e-12) * (1 - np.exp(np.clip(-betas[j] * (T - jump), -100, 100)))
    else:
        ## Unpack parameters for 10-parameter model (delayed scaled exponential)
        lambda0, theta0_alpha, eta_alpha, csi_alpha, theta0_beta, eta_beta, csi_beta, theta0_delta, eta_delta, csi_delta = np.exp(params)
        ## Calculate hierarchical parameters
        alphas = inhibitory_hawkes_intensity(jumps, jumps, theta0_alpha, eta_alpha, csi_alpha)
        betas = inhibitory_hawkes_intensity(jumps, jumps, theta0_beta, eta_beta, csi_beta)
        deltas = inhibitory_hawkes_intensity(jumps, jumps, theta0_delta, eta_delta, csi_delta)
        ## Calculate integral for each jump (Equation 3 from the paper)
        integral = 0.0
        for j, jump in enumerate(jumps):
            if T > jump:
                time_diff = T - jump
                if time_diff <= deltas[j]:
                    # I_j = alpha(tau_j) / (2 * delta(tau_j)) * (T - tau_j)^2
                    integral += alphas[j] / (2 * deltas[j] + 1e-12) * (time_diff ** 2)
                else:
                    # I_j = alpha(tau_j) * delta(tau_j) / 2 + alpha(tau_j) / beta(tau_j) * [1 - exp(-beta(tau_j) * (T - tau_j - delta(tau_j)))]
                    integral += alphas[j] * deltas[j] / 2 + alphas[j] / (betas[j] + 1e-12) * (1 - np.exp(np.clip(-betas[j] * (time_diff - deltas[j]), -100, 100)))
    ## Subtract integrals over censored (blank) periods if observation intervals are provided
    if observation_intervals is not None:
        censored_integral = 0.0
        for r in range(len(observation_intervals)):
            a_r, b_r = observation_intervals[r]
            # Calculate integral over [a_r, b_r] for the baseline
            censored_integral += lambda0 * (b_r - a_r)
            # Calculate integral for each jump's contribution
            if len(params) == 7:
                for j, jump in enumerate(jumps):
                    if b_r > jump:
                        # Contribution to [a_r, b_r]
                        # Upper limit contribution
                        upper = alphas[j] / (betas[j] + 1e-12) * (1 - np.exp(np.clip(-betas[j] * (b_r - jump), -100, 100))) if b_r > jump else 0.0
                        # Lower limit contribution
                        lower = alphas[j] / (betas[j] + 1e-12) * (1 - np.exp(np.clip(-betas[j] * (a_r - jump), -100, 100))) if a_r > jump else 0.0
                        censored_integral += upper - lower
            else:
                for j, jump in enumerate(jumps):
                    if b_r > jump:
                        # Contribution to [a_r, b_r] - upper limit
                        if b_r <= jump:
                            upper = 0.0
                        else:
                            time_diff_upper = b_r - jump
                            if time_diff_upper <= deltas[j]:
                                upper = alphas[j] / (2 * deltas[j] + 1e-12) * (time_diff_upper ** 2)
                            else:
                                upper = alphas[j] * deltas[j] / 2 + alphas[j] / (betas[j] + 1e-12) * (1 - np.exp(np.clip(-betas[j] * (time_diff_upper - deltas[j]), -100, 100)))
                        # Contribution to [a_r, b_r] - lower limit
                        if a_r <= jump:
                            lower = 0.0
                        else:
                            time_diff_lower = a_r - jump
                            if time_diff_lower <= deltas[j]:
                                lower = alphas[j] / (2 * deltas[j] + 1e-12) * (time_diff_lower ** 2)
                            else:
                                lower = alphas[j] * deltas[j] / 2 + alphas[j] / (betas[j] + 1e-12) * (1 - np.exp(np.clip(-betas[j] * (time_diff_lower - deltas[j]), -100, 100)))                        
                        censored_integral += upper - lower
        # Adjust the total integral: instead of [0, T], we only integrate over observation intervals
        integral = censored_integral
    ## Calculate negative log-likelihood
    if isinstance(event_times, dict):
        # For multiple sequences (mixture model)
        nll_val = 0.0 if sum_values else np.zeros(n)
        for i in range(n):
            # Sum of log intensities at event times
            log_sum = np.sum(np.log(lik1[i] + 1e-12))
            # Weighted negative log-likelihood
            if sum_values:
                nll_val -= weights[i] * (log_sum - integral - lambda0 * T)
            else:
                nll_val[i] -= weights[i] * (log_sum - integral - lambda0 * T)
    else:
        # For single sequence
        log_sum = np.sum(np.log(lik1 + 1e-12))
        nll_val = -log_sum + lambda0 * T + integral
    # Return the (weighted) negative log-likelihood value
    return nll_val