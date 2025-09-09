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

## Maximum likelihood estimation of the parameters of the entire process
def negative_log_likelihood(params, time_series, jumps):
    ## Check parameter values
    if not isinstance(params, (list, np.ndarray)):
        raise TypeError("Parameters must be a list or numpy array.")
    if not isinstance(time_series, (list, np.ndarray)):
        raise TypeError("Time series must be a list or numpy array.")
    if not isinstance(jumps, (list, np.ndarray)):
        raise TypeError("Jumps must be a list or numpy array.")
    if len(params) != 7:
        raise ValueError("Expected 7 parameters, got {}".format(len(params)))
    ## Convert inputs to numpy arrays to avoid type conflicts
    params = np.clip(params, -100, 100)
    time_series = np.array(time_series)
    jumps = np.array(jumps)
    ## Model parameters are only obtained after exponentiation
    lambda0, theta0_alpha, eta_alpha, csi_alpha, theta0_beta, eta_beta, csi_beta = np.exp(params)
    ## Calculate the model parameters hierarchically
    alphas = inhibitory_hawkes_intensity(jumps, jumps, theta0_alpha, eta_alpha, csi_alpha)
    betas = inhibitory_hawkes_intensity(jumps, jumps, theta0_beta, eta_beta, csi_beta)
    ## Time grid for the intensity function
    time_grid = np.arange(len(time_series))
    ints = np.ones_like(time_grid) * lambda0
    # Calculate the intensity function
    for j, jump in enumerate(jumps):
        ints += alphas[j] * np.exp(np.clip(-betas[j] * (time_grid - jump), -100, 100)) * np.heaviside(time_grid - jump, 1) ## Returns 1 if time_grid >= jump, else 0
    ## Loss function (negative log-likelihood / squared error loss)
    loss = np.sum((ints - time_series) ** 2)
    return loss

## Maximum likelihood estimation of the parameters of the entire process
def negative_log_likelihood_full(params, time_series, jumps):
    ## Check parameter values
    if not isinstance(params, (list, np.ndarray)):
        raise TypeError("Parameters must be a list or numpy array.")
    if not isinstance(time_series, (list, np.ndarray)):
        raise TypeError("Time series must be a list or numpy array.")
    if not isinstance(jumps, (list, np.ndarray)):
        raise TypeError("Jumps must be a list or numpy array.")
    if len(params) != 10:
        raise ValueError("Expected 10 parameters, got {}".format(len(params)))
    ## Convert inputs to numpy arrays to avoid type conflicts
    params = np.clip(params, -100, 100)
    time_series = np.array(time_series)
    jumps = np.array(jumps)
    ## Model parameters are only obtained after exponentiation
    lambda0, theta0_alpha, eta_alpha, csi_alpha, theta0_beta, eta_beta, csi_beta, theta0_delta, eta_delta, csi_delta = np.exp(params)
    ## Calculate the model parameters hierarchically
    alphas = inhibitory_hawkes_intensity(jumps, jumps, theta0_alpha, eta_alpha, csi_alpha)
    betas = inhibitory_hawkes_intensity(jumps, jumps, theta0_beta, eta_beta, csi_beta)
    deltas = inhibitory_hawkes_intensity(jumps, jumps, theta0_delta, eta_delta, csi_delta)
    ## Time grid for the intensity function
    time_grid = np.arange(len(time_series))
    ints = np.ones_like(time_grid) * lambda0
    # Calculate the intensity function
    for j, jump in enumerate(jumps):
        c1 = alphas[j] / (deltas[j] + 1e-12) * np.clip(time_grid - jump, 0, deltas[j]) * (1 - np.heaviside(time_grid - jump - deltas[j], 1))
        c2 = np.heaviside(time_grid - jump - deltas[j], 1) * alphas[j] * np.exp(np.clip(-betas[j] * (time_grid - jump - deltas[j]), -100, 100))  
        ints += (c1 + c2) * np.heaviside(time_grid - jump, 1) ## Returns 1 if time_grid >= jump, else 0
    ## Loss function (negative log-likelihood / squared error loss)
    loss = np.sum((ints - time_series) ** 2)
    return loss

## Minimize the least squares loss function around each inhibitory intensity
def inhibitory_loss(params, obs, jumps):
    ## Check parameter values
    if not isinstance(params, (list, np.ndarray)):
        raise TypeError("Parameters must be a list or numpy array.")
    if not isinstance(obs, (list, np.ndarray)):
        raise TypeError("Observed values must be a list or numpy array.")
    if not isinstance(jumps, (list, np.ndarray)):
        raise TypeError("Jumps must be a list or numpy array.")
    if len(params) != 3:
        raise ValueError("Expected 3 parameters, got {}".format(len(params)))
    ## Convert inputs to numpy arrays to avoid type conflicts
    params = np.clip(params, -100, 100)
    obs = np.array(obs)
    jumps = np.array(jumps)
    ## Model parameters are only obtained after exponentiation
    theta0, eta, xi = np.exp(params)
    ## Calculate the predicted intensity
    preds = inhibitory_hawkes_intensity(jumps, jumps, theta0, eta, xi)
    ## Loss function
    loss = np.sum((obs - preds) ** 2)
    return loss

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
def nll(params, time_series, jumps, weights):
    ## Check parameter values
    if not isinstance(params, (list, np.ndarray)):
        raise TypeError("Parameters must be a list or numpy array.")
    if not isinstance(time_series, np.ndarray):
        raise TypeError("Time series must be a numpy array with shape (n, T).")
    if time_series.ndim != 2:
        raise ValueError("Time series must be a 2D array.")
    if not isinstance(jumps, (list, np.ndarray)):
        raise TypeError("Jumps must be a list or numpy array.")
    if weights is None:
        weights = np.ones_like(time_series)
    else:
        if not isinstance(weights, (list, np.ndarray)):
            raise TypeError("Weights must be a list or numpy array.")
        if weights.shape != time_series.shape:
            raise ValueError("Weights must be of the same shape as time_series.")
    if len(params) != 7 and len(params) != 10:
        raise ValueError("Expected 7 or 10 parameters, got {}".format(len(params)))
    ## Convert inputs to numpy arrays to avoid type conflicts
    params = np.clip(params, -100, 100)
    n, T = time_series.shape
    jumps = np.array(jumps)
    ## Obtain predicted curve
    ints = calculate_intensity(params, np.arange(T), jumps)
    ## Calculate squared error loss function, reweighted
    loss = 0
    for i in range(n):
        loss += np.sum(weights[i] * ((ints - time_series[i]) ** 2)) 
    return loss / 2