#!/usr/bin/env python3

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from scipy.optimize import minimize
## Import functions from file hhf.py
from .hhf import inhibitory_loss, negative_log_likelihood

class HierarchicalHakwesMixtureEM:
    """
    EM for model: y_it | z_i=k ~ Normal[f_k(t; theta_k), sigma2_{k,t}],
    where f_k is a Hierarchical Hawkes function.
    For fitting, a function fit_fn must be provided to fit parameters theta_k from *weighted* observations
    """
    def __init__(self, K, fit_fn=None, predict_fn=None, jumps=None, jumps_alpha=None, 
                    delay=False, max_iter=200, tol=1e-6, verbose=True):
        """
        K          : number of clusters
        fit_fn     : fit_fn(data_matrix, weight_matrix, jumps, init) -> theta (as part of result object)
        predict_fn : predict_fn(theta, times) -> mean_response_vector
        """
        # Check if K is an integer
        if not isinstance(K, int):
            raise ValueError("K must be an integer.")
        self.K = int(K)
        # Check if delay is a boolean
        if not isinstance(delay, bool):
            raise ValueError("Delay must be a boolean (True/False).")
        self.delay = delay
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
        if fit_fn is None or predict_fn is None:
            raise ValueError("Both fit_fn and predict_fn must be provided.")
        # Initialize parameters
        self.fit_fn = fit_fn
        self.predict_fn = predict_fn
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.verbose = bool(verbose)
        # Placeholders to be filled at fitting stage
        self.pi = None          # (K,)
        self.theta = None       # List of process parameters
        self.sigma2 = None      # (K, T)
        self.loglik_trace = []
        # Return output if verbose
        if self.verbose:
            print('Model Summary:')
            print(f'  - Number of clusters (K): {self.K}')
            print(f'  - Jump locations: {self.jumps}')
            if self.jumps_alpha is not None:
                print(f'  - Jump locations (alpha): {self.jumps_alpha}')
            print(f'  - Delay in jumps: {self.delay}')

    ## Initialisation of parameters
    def init_params(self, Y, init_method='kmeans', offset=1e-12):
        n, T = Y.shape
        K = self.K
        if init_method == 'kmeans':
            # Cluster raw series
            kmeans = KMeans(n_clusters=K).fit(Y)
            labels = kmeans.labels_
            r = 0.5 / (K - 1) * np.ones((n, K))
            r[np.arange(n), labels] = 0.5
        else:
            # Random soft assignments from Dirichlet distribution
            r = np.random.dirichlet(np.ones(K), size=n)
        # Initialise pi
        self.pi = (r.sum(axis=0) + offset) / n # Add small constant to avoid zeros
        # Initialise variances from cluster residuals from cluster means
        self.sigma2 = np.zeros((K, T))
        cluster_means = np.zeros((K, T))
        for k in range(K):
            wk = r[:, k]
            if wk.sum() > 0:
                cluster_means[k] = (wk @ Y) / wk.sum()
            else:
                cluster_means[k] = Y.mean(axis=0)
            self.sigma2[k] = (((Y - cluster_means[k]) ** 2).T @ wk) / wk.sum() + offset
        # Initialize mean-function params
        self.theta = []
        for k in range(K):
            if self.delay:
                lambda0 = np.mean(cluster_means[k][:self.jumps_alpha[0]])
                alphas0 = np.diff(cluster_means[k])[self.jumps_alpha - 1]
                if np.any(alphas0 <= 0):
                    ## Select the maximum differences observed
                    alphas0 = np.sort(np.diff(cluster_means[k]))[::-1][:len(self.jumps_alpha)]
                theta0_alpha0 = alphas0[0]
                eta_alpha0 = np.abs(alphas0[1] - alphas0[0])
                csi_alpha0 = 0.1  # Initial guess for csi
                params = np.log([theta0_alpha0, eta_alpha0, csi_alpha0])
                res_alpha = minimize(inhibitory_loss, params, args=(alphas0, self.jumps_alpha), method='L-BFGS-B')
                params_init = np.log([lambda0] + list(np.exp(res_alpha.x)) + [1.5, 0.5, 0.1])
                # Minimize the negative log-likelihood
                result_init = minimize(negative_log_likelihood, params_init, args=(cluster_means[k], self.jumps_alpha), method='L-BFGS-B')
                ## Optimize the parameters
                params = np.log(list(np.exp(result_init.x)) + [1.5, 0.5, 0.25])
            else:
                # Initial parameter guess
                lambda0 = np.mean(cluster_means[k][:self.jumps[0]])
                alphas0 = np.diff(cluster_means[k])[self.jumps - 1]
                if np.any(alphas0 <= 0):
                    ## Select the maximum differences observed
                    alphas0 = np.sort(np.diff(cluster_means[k]))[::-1][:len(self.jumps)]
                theta0_alpha0 = alphas0[0]
                eta_alpha0 = np.abs(alphas0[1] - alphas0[0])
                csi_alpha0 = 0.1  # Initial guess for csi
                params = np.log([theta0_alpha0, eta_alpha0, csi_alpha0])
                res_alpha = minimize(inhibitory_loss, params, args=(alphas0, self.jumps), method='L-BFGS-B')
                params = np.log([lambda0] + list(np.exp(res_alpha.x)) + [0.35, 0.2, 0.1])
            ## Parameter initialisation
            theta = self.fit_fn(y=cluster_means[k].reshape(1,-1), w=np.ones(T).reshape(1,-1), jumps=self.jumps, init=params).x
            self.theta.append(theta)

    ## Calculate responsibilities at E-step
    def E_step(self, Y):
        n, T = Y.shape
        K = self.K
        log_resp = np.zeros((n, K))
        # compute log p(y_i | k) + log pi_k
        for k in range(K):
            # Use the predict function to obtain the mean response
            mu_k = self.predict_fn(params=self.theta[k], t=np.arange(T), jumps=self.jumps)  # length T
            log_det = -0.5 * np.sum(np.log(2 * np.pi * self.sigma2[k]))
            # Quadratic term per series - ((Y - mu)^2 / sigma2) summed over all t
            q = -0.5 * (((Y - mu_k) ** 2) / self.sigma2[k]).sum(axis=1)
            log_prob = log_det + q
            log_resp[:, k] = np.log(self.pi[k] + 1e-12) + log_prob
        # Normalize probabilities with logsumexp
        lse = logsumexp(log_resp, axis=1, keepdims=True)
        resp = np.exp(log_resp - lse)
        return resp, lse.sum()  # responsibilities and log-likelihood contribution

    # Fit the mixture model
    def fit(self, Y, init_method='kmeans', offset=1e-12):
        """
        Fit the mixture.
        Y           : (n, T) data matrix
        init_method : 'kmeans', alternatively random Dirichlet initialisation is used
        predict_fn : prediction function of the form predict_fn(theta, time_indices) -> mean_vector
        """
        Y = np.asarray(Y)
        n, T = Y.shape
        # Initialise parameters
        resp = self.init_params(Y, init_method=init_method)
        # Initialise previous value of likelihood
        prev_ll = -np.inf
        for it in range(self.max_iter):
            # E-step
            resp, ll = self.E_step(Y)
            self.loglik_trace.append(ll)
            # M-step
            S_k = resp.sum(axis=0) + offset # (K,)
            self.pi = S_k / n
            # Update process parameters
            for k in range(self.K):
                # Calculate weights and pass to the fitting function to estimate parameters
                w_k = np.outer(resp[:, k], 1 / self.sigma2[k]) ## (n, T)
                theta_prev = np.copy(self.theta[k])
                ## First option for fitting process
                fit1 = self.fit_fn(y=Y, w=w_k, jumps=self.jumps, init=theta_prev)
                ## Second option for fitting process
                mu_prev = self.predict_fn(params=theta_prev, t=np.arange(T), jumps=self.jumps)
                try:
                    if self.delay:
                        lambda0 = np.mean(mu_prev[:self.jumps_alpha[0]])
                        alphas0 = np.diff(mu_prev)[self.jumps_alpha - 1]
                        if np.any(alphas0 <= 0):
                            ## Select the maximum differences observed
                            alphas0 = np.sort(np.diff(mu_prev))[::-1][:len(self.jumps_alpha)]
                        theta0_alpha0 = alphas0[0]
                        eta_alpha0 = np.abs(alphas0[1] - alphas0[0])
                        csi_alpha0 = 0.1  # Initial guess for csi
                        params = np.log([theta0_alpha0, eta_alpha0, csi_alpha0])
                        res_alpha = minimize(inhibitory_loss, params, args=(alphas0, self.jumps_alpha), method='L-BFGS-B')
                        params_init = np.log([lambda0] + list(np.exp(res_alpha.x)) + [1.5, 0.5, 0.1])
                        # Minimize the negative log-likelihood
                        result_init = minimize(negative_log_likelihood, params_init, args=(mu_prev, self.jumps_alpha), method='L-BFGS-B')
                        ## Optimize the parameters
                        params = np.log(list(np.exp(result_init.x)) + [1.5, 0.5, 0.25])
                    else:
                        # Initial parameter guess
                        lambda0 = np.mean(mu_prev[:self.jumps[0]])
                        alphas0 = np.diff(mu_prev)[self.jumps - 1]
                        if np.any(alphas0 <= 0):
                            ## Select the maximum differences observed
                            alphas0 = np.sort(np.diff(mu_prev))[::-1][:len(self.jumps)]
                        theta0_alpha0 = alphas0[0]
                        eta_alpha0 = np.abs(alphas0[1] - alphas0[0])
                        csi_alpha0 = 0.1  # Initial guess for csi
                        params = np.log([theta0_alpha0, eta_alpha0, csi_alpha0])
                        res_alpha = minimize(inhibitory_loss, params, args=(alphas0, self.jumps), method='L-BFGS-B')
                        params = np.log([lambda0] + list(np.exp(res_alpha.x)) + [0.35, 0.2, 0.1])
                    fit2 = self.fit_fn(y=Y, w=w_k, jumps=self.jumps, init=params)
                    ## Pick best outcome between fit1 and fit2
                    if fit1.success and fit2.success:
                        if fit1.fun < fit2.fun:
                            self.theta[k] = fit1.x
                        else:
                            self.theta[k] = fit2.x
                    else:
                        self.theta[k] = fit2.x if fit2.success else fit1.x
                except:
                    self.theta[k] = fit1.x
                # Predict process with the updated parameters
                mu_k = self.predict_fn(params=self.theta[k], t=np.arange(T), jumps=self.jumps)
                # Update variances sigma2_{k,t}
                num = (resp[:, k][:, None] * (Y - mu_k[None, :]) ** 2).sum(axis=0)
                self.sigma2[k] = np.maximum(num / (S_k[k] + offset), 1e-2)
            # Check convergence
            if self.verbose:
                print(f"Iteration: {it:3d}\t Log-likelihood: {ll:.6f}\t\t\t", end='\r')
            if np.abs(ll - prev_ll) < self.tol * max(1.0, np.abs(prev_ll)):
                break
            prev_ll = ll
        print()
        print(f"Final log-likelihood: {ll:.6f}")