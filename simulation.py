"""
Numerical simulation engine for the Ornstein-Uhlenbeck (OU) process.
Implements the Euler-Maruyama scheme and computes empirical and theoretical statistics.
"""

import numpy as np

def simulate_ou_process(theta, mu, sigma, t_max, steps, num_paths, x0, x0_mode='fixed'):
    """
    Simulates multiple trajectories of the Ornstein-Uhlenbeck process using Euler-Maruyama integration.
    dX_t = theta * (mu - X_t) * dt + sigma * dW_t

    Parameters:
    -----------
    theta : float
        Mean-reversion rate.
    mu : float
        Long-term equilibrium level.
    sigma : float
        Volatility / diffusion coefficient.
    t_max : float
        Total time horizon.
    steps : int
        Number of time discretization steps.
    num_paths : int
        Number of independent trajectories to generate.
    x0 : float
        Initial value (ignored if x0_mode is 'stationary').
    x0_mode : str
        Initial condition mode: 'fixed' or 'stationary'.

    Returns:
    --------
    time_grid : np.ndarray
        1D array of shape (steps + 1,) containing discretized times.
    paths : np.ndarray
        2D array of shape (num_paths, steps + 1) containing simulated trajectories.
    mean_path : np.ndarray
        1D array of shape (steps + 1,) containing empirical mean across all paths at each step.
    sd_path : np.ndarray
        1D array of shape (steps + 1,) containing empirical standard deviation at each step.
    """
    dt = t_max / steps
    time_grid = np.linspace(0, t_max, steps + 1)
    
    # Initialize paths array
    paths = np.zeros((num_paths, steps + 1))
    
    # 1. Set initial condition
    if x0_mode == 'stationary' and theta > 0:
        stationary_sd = np.sqrt(sigma**2 / (2.0 * theta))
        paths[:, 0] = np.random.normal(mu, stationary_sd, size=num_paths)
    else:
        paths[:, 0] = x0
        
    # 2. Euler-Maruyama integration
    # pre-generate Gaussian random increments for speed
    # Wiener process increments: dW = Z * sqrt(dt)
    sqrt_dt = np.sqrt(dt)
    
    for i in range(steps):
        x = paths[:, i]
        # Generate independent standard normals for each path at this step
        z = np.random.normal(0, 1, size=num_paths)
        # SDE update: dX_t = theta * (mu - X_t) * dt + sigma * dW_t
        dx = theta * (mu - x) * dt + sigma * sqrt_dt * z
        paths[:, i + 1] = x + dx
        
    # 3. Calculate empirical ensemble statistics
    mean_path = np.mean(paths, axis=0)
    sd_path = np.std(paths, axis=0, ddof=1) if num_paths > 1 else np.zeros(steps + 1)
    
    return time_grid, paths, mean_path, sd_path

def compute_empirical_acf(paths, max_lag_steps):
    """
    Computes the empirical autocorrelation function (ACF) of the simulated trajectories,
    averaged across all independent paths.

    Parameters:
    -----------
    paths : np.ndarray
        2D array of shape (num_paths, steps + 1) containing trajectories.
    max_lag_steps : int
        Maximum lag step index to calculate.

    Returns:
    --------
    acf_ensemble : np.ndarray
        1D array of shape (max_lag_steps + 1,) containing the average empirical ACF.
    """
    num_paths, steps_plus_1 = paths.shape
    max_lag = min(max_lag_steps, steps_plus_1 - 2)
    
    acf_ensemble = np.zeros(max_lag + 1)
    
    for p in range(num_paths):
        path = paths[p]
        mean = np.mean(path)
        var = np.var(path)
        
        if var == 0:
            acf_ensemble += 1.0
            continue
            
        deviations = path - mean
        # Fast ACF calculation for this path
        for lag in range(max_lag + 1):
            if lag == 0:
                acf_ensemble[0] += 1.0
            else:
                acf_ensemble[lag] += np.mean(deviations[:-lag] * deviations[lag:]) / var
                
    acf_ensemble /= num_paths
    return acf_ensemble

def get_theoretical_transition_stats(t, x0, theta, mu, sigma):
    """
    Computes theoretical mean and standard deviation at time t given initial condition x0.
    Handles mean-reverting (theta > 0), mean-fleeing (theta < 0), and random walk (theta = 0) cases.
    """
    if t <= 1e-9:
        return x0, 0.0
        
    if theta > 0 or theta < 0:
        exp_theta_t = np.exp(-theta * t)
        mean_t = x0 * exp_theta_t + mu * (1.0 - exp_theta_t)
        var_t = (sigma**2 / (2.0 * theta)) * (1.0 - np.exp(-2.0 * theta * t))
    else:
        # theta = 0 (Arithmetic Brownian Motion)
        mean_t = x0
        var_t = (sigma**2) * t
        
    if var_t < 0:
        var_t = 0.0
        
    return mean_t, np.sqrt(var_t)

def get_theoretical_transition_pdf(x_grid, t, x0, theta, mu, sigma):
    """
    Computes theoretical transition density PDF values on x_grid at time t.
    """
    mean_t, sd_t = get_theoretical_transition_stats(t, x0, theta, mu, sigma)
    
    if sd_t <= 1e-5:
        # If t is near 0, standard deviation is near 0.
        # Return a narrow Gaussian to approximate Dirac delta
        sd_t = 1e-3
        
    exponent = -0.5 * ((x_grid - mean_t) / sd_t) ** 2
    pdf = (1.0 / (sd_t * np.sqrt(2.0 * np.pi))) * np.exp(exponent)
    return pdf

def get_theoretical_stationary_pdf(x_grid, theta, mu, sigma):
    """
    Computes theoretical stationary density PDF values on x_grid.
    Returns None if stationary distribution does not exist (theta <= 0).
    """
    if theta <= 0:
        return None
        
    var_stationary = sigma**2 / (2.0 * theta)
    sd_stationary = np.sqrt(var_stationary)
    
    exponent = -0.5 * ((x_grid - mu) / sd_stationary) ** 2
    pdf = (1.0 / (sd_stationary * np.sqrt(2.0 * np.pi))) * np.exp(exponent)
    return pdf
