## Utility functions for hierarchical Hawkes functions
import numpy as np

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
    params = np.array(params)
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
        ints += alphas[j] * np.exp(-betas[j] * (time_grid - jump)) * np.heaviside(time_grid - jump, 1) ## Returns 1 if time_grid >= jump, else 0
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
    params = np.array(params)
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
        c1 = alphas[j] / deltas[j] * np.clip(time_grid - jump, 0, deltas[j]) * (1 - np.heaviside(time_grid - jump - deltas[j], 1))
        c2 = np.heaviside(time_grid - jump - deltas[j], 1) * alphas[j] * np.exp(-betas[j] * (time_grid - jump - deltas[j]))  
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
    params = np.array(params)
    obs = np.array(obs)
    jumps = np.array(jumps)
    ## Model parameters are only obtained after exponentiation
    theta0, eta, xi = np.exp(params)
    ## Calculate the predicted intensity
    preds = inhibitory_hawkes_intensity(jumps, jumps, theta0, eta, xi)
    ## Loss function
    loss = np.sum((obs - preds) ** 2)
    return loss