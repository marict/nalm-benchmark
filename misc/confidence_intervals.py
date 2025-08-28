#!/usr/bin/env python3
"""
Statistical confidence intervals utility module for NALM benchmark evaluation.

Provides confidence interval calculations for various metrics using appropriate
statistical distributions as per academic paper methodology.
"""

import math
from typing import List, Tuple, Optional

import numpy as np
from scipy import stats


def binomial_confidence_interval(
    successes: int, 
    trials: int, 
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for success rates using binomial distribution.
    
    Args:
        successes: Number of successful trials (e.g., number of grokking runs)
        trials: Total number of trials (e.g., total number of seeds tested)
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound) for the success rate
    """
    if trials == 0:
        return (0.0, 0.0)
    
    alpha = 1 - confidence
    
    # Use Wilson score interval for better small-sample behavior
    p_hat = successes / trials
    z = stats.norm.ppf(1 - alpha/2)
    
    denominator = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denominator
    half_width = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * trials)) / trials) / denominator
    
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    
    return (lower, upper)


def gamma_confidence_interval(
    values: List[float], 
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for convergence times using gamma distribution.
    
    Args:
        values: List of convergence times (e.g., steps to grok)
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound) for the mean convergence time
    """
    if not values or len(values) == 0:
        return (float('inf'), float('inf'))
    
    values = np.array(values)
    n = len(values)
    
    # Fit gamma distribution using method of moments
    sample_mean = np.mean(values)
    sample_var = np.var(values, ddof=1)
    
    if sample_var <= 0 or sample_mean <= 0:
        # Fallback to normal distribution if gamma assumptions are violated
        return normal_confidence_interval(values, confidence)
    
    # Method of moments estimators for gamma distribution
    # shape = mean^2 / variance, scale = variance / mean
    shape_est = sample_mean**2 / sample_var
    scale_est = sample_var / sample_mean
    
    # Confidence interval for the mean of gamma distribution
    alpha = 1 - confidence
    
    # For gamma distribution, the sample mean follows a gamma distribution
    # with shape = n * shape_est and scale = scale_est / n
    ci_shape = n * shape_est
    ci_scale = scale_est / n
    
    lower = stats.gamma.ppf(alpha/2, a=ci_shape, scale=ci_scale)
    upper = stats.gamma.ppf(1 - alpha/2, a=ci_shape, scale=ci_scale)
    
    return (lower, upper)


def beta_confidence_interval(
    values: List[float], 
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for sparsity errors using beta distribution.
    
    Args:
        values: List of sparsity error values (bounded between 0 and 1)
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound) for the mean sparsity error
    """
    if not values or len(values) == 0:
        return (0.0, 1.0)
    
    values = np.array(values)
    values = np.clip(values, 1e-10, 1 - 1e-10)  # Avoid boundary issues
    
    n = len(values)
    sample_mean = np.mean(values)
    sample_var = np.var(values, ddof=1)
    
    if sample_var <= 0:
        # No variance, return point estimate
        return (sample_mean, sample_mean)
    
    # Method of moments estimators for beta distribution
    # alpha = mean * (mean * (1 - mean) / var - 1)
    # beta = (1 - mean) * (mean * (1 - mean) / var - 1)
    var_scale = sample_mean * (1 - sample_mean) / sample_var
    
    if var_scale <= 1:
        # Fallback to normal distribution if beta assumptions are violated
        return normal_confidence_interval(values, confidence, lower_bound=0.0, upper_bound=1.0)
    
    alpha_est = sample_mean * (var_scale - 1)
    beta_est = (1 - sample_mean) * (var_scale - 1)
    
    # For beta distribution, we can use bootstrap or normal approximation
    # Using normal approximation for the mean
    std_error = math.sqrt(sample_var / n)
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha/2)
    
    lower = max(0.0, sample_mean - z * std_error)
    upper = min(1.0, sample_mean + z * std_error)
    
    return (lower, upper)


def normal_confidence_interval(
    values: List[float], 
    confidence: float = 0.95,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None
) -> Tuple[float, float]:
    """
    Calculate confidence interval using normal distribution (fallback method).
    
    Args:
        values: List of numerical values
        confidence: Confidence level (default 0.95 for 95% CI)
        lower_bound: Optional lower bound to clip results
        upper_bound: Optional upper bound to clip results
        
    Returns:
        Tuple of (lower_bound, upper_bound) for the mean
    """
    if not values or len(values) == 0:
        return (float('-inf'), float('inf'))
    
    values = np.array(values)
    n = len(values)
    sample_mean = np.mean(values)
    
    if n == 1:
        return (sample_mean, sample_mean)
    
    sample_std = np.std(values, ddof=1)
    std_error = sample_std / math.sqrt(n)
    
    alpha = 1 - confidence
    # Use t-distribution for small samples
    if n <= 30:
        t_val = stats.t.ppf(1 - alpha/2, df=n-1)
    else:
        t_val = stats.norm.ppf(1 - alpha/2)
    
    lower = sample_mean - t_val * std_error
    upper = sample_mean + t_val * std_error
    
    if lower_bound is not None:
        lower = max(lower, lower_bound)
    if upper_bound is not None:
        upper = min(upper, upper_bound)
    
    return (lower, upper)


def format_confidence_interval(
    lower: float, 
    upper: float, 
    point_estimate: Optional[float] = None,
    precision: int = 4
) -> str:
    """
    Format confidence interval for display.
    
    Args:
        lower: Lower bound of confidence interval
        upper: Upper bound of confidence interval  
        point_estimate: Optional point estimate to include
        precision: Number of decimal places
        
    Returns:
        Formatted string representation
    """
    if point_estimate is not None:
        return f"{point_estimate:.{precision}f} [{lower:.{precision}f}, {upper:.{precision}f}]"
    else:
        return f"[{lower:.{precision}f}, {upper:.{precision}f}]"


# Example usage and test functions
if __name__ == "__main__":
    # Test binomial confidence interval
    print("Testing binomial CI:")
    successes, trials = 18, 25
    ci = binomial_confidence_interval(successes, trials)
    success_rate = successes / trials
    print(f"Success rate: {format_confidence_interval(ci[0], ci[1], success_rate)}")
    
    # Test gamma confidence interval  
    print("\nTesting gamma CI:")
    convergence_times = [100, 150, 200, 180, 120, 300, 250]
    ci = gamma_confidence_interval(convergence_times)
    mean_time = np.mean(convergence_times)
    print(f"Mean convergence time: {format_confidence_interval(ci[0], ci[1], mean_time)}")
    
    # Test beta confidence interval
    print("\nTesting beta CI:")
    sparsity_errors = [0.1, 0.05, 0.15, 0.08, 0.12, 0.03, 0.09]
    ci = beta_confidence_interval(sparsity_errors)
    mean_sparsity = np.mean(sparsity_errors)
    print(f"Mean sparsity error: {format_confidence_interval(ci[0], ci[1], mean_sparsity)}")