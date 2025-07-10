"""
Statistical Analysis Functions for Glioblastoma Project

This module contains statistical functions extracted from ci_mann_whitney_new_copy.py
for analyzing model predictions and calculating confidence intervals.

Author: Glioblastoma Research Team
Date: July 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency
from statsmodels.nonparametric.kde import KDEUnivariate
from typing import Tuple, Optional, Union

# =============================================================================
# COHORT ASSIGNMENT FUNCTIONS
# =============================================================================

def assign_cohort(age):
    """
    Assign age-based cohort groups for analysis.
    Original function from ci_mann_whitney_new_copy.py
    
    Args:
        age: Age in years
        
    Returns:
        str: Cohort group ('Young', 'Adults', 'Older Adults')
    """
    if age < 27:
        return 'Young'
    elif 27 <= age <= 59:
        return 'Adults'
    else:
        return 'Older Adults'

def asignar_cohorte(age):
    """
    Function to assign cohorts by age.
    Original function from test_power_mannwhitney.py
    
    Args:
        age: Age in years
        
    Returns:
        str: Cohort group ('Young', 'Adults', 'Older Adults')
    """
    if age < 27:
        return 'Young'
    elif 27 <= age <= 59:
        return 'Adults'
    else:
        return 'Older Adults'

# =============================================================================
# CONFIDENCE INTERVAL FUNCTIONS
# =============================================================================

def calculate_ci_bootstrap(data, n_bootstrap=1000, alpha=0.05):
    """
    Calculate bootstrap confidence intervals for the mean.
    Original function from ci_mann_whitney_new_copy.py
    
    Args:
        data: Input data
        n_bootstrap: Number of bootstrap samples (default: 1000)
        alpha: Significance level (default: 0.05 for 95% CI)
        
    Returns:
        tuple: (lower_bound, upper_bound, mean) or (None, None, mean) if insufficient data
    """
    if len(data) < 2:
        return None, None, np.mean(data) if len(data) == 1 else None
    bootstrap_means = [np.random.choice(data, size=len(data), replace=True).mean() for _ in range(n_bootstrap)]
    lower_bound = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper_bound = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    return lower_bound, upper_bound, np.mean(data)

def calculate_ci_normal(data, alpha=0.05):
    """
    Calculate confidence intervals assuming normal distribution.
    Original function from ci_mann_whitney_new_copy.py
    
    Args:
        data: Input data
        alpha: Significance level (default: 0.05 for 95% CI)
        
    Returns:
        tuple: (lower_bound, upper_bound, mean) or (None, None, mean) if insufficient data
    """
    if len(data) < 2:
        return None, None, np.mean(data) if len(data) == 1 else None
    mean = np.mean(data)
    std_error = np.std(data, ddof=1) / np.sqrt(len(data))
    z = 1.96  # For 95% confidence
    lower_bound = mean - z * std_error
    upper_bound = mean + z * std_error
    return lower_bound, upper_bound, mean

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def create_kde_comparison_plot(df, cohorts=['Young', 'Adults', 'Older Adults'], 
                              figsize=(14, 5), save_path=None):
    """
    Create KDE plots comparing correct vs incorrect predictions by age cohort.
    Based on the plotting code from ci_mann_whitney_new_copy.py
    
    Args:
        df: DataFrame with columns 'Age_Group', 'Correct_Prediction', 'Age'
        cohorts: List of cohort names to plot
        figsize: Figure size tuple
        save_path: Path to save the figure (optional)
        
    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    # Set up plotting style
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 19
    
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    
    print("=== CONFIDENCE INTERVALS - COMPARISON METHODS (95%) ===")
    print()
    
    for i, cohort in enumerate(cohorts):
        cohort_data = df[df['Age_Group'] == cohort]
        correct = cohort_data[cohort_data['Correct_Prediction'] == 1]['Age']
        incorrect = cohort_data[cohort_data['Correct_Prediction'] == 0]['Age']

        if not correct.empty:
            # Distribution for Correct predictions (KDE)
            kde_c = KDEUnivariate(correct)
            kde_c.fit(bw="scott")
            x_c = np.linspace(correct.min() - 5, correct.max() + 5, 200)
            y_c = kde_c.evaluate(x_c)
            ci_lower_c, ci_upper_c, mean_c = calculate_ci_bootstrap(correct)
            ci_lower_c_norm, ci_upper_c_norm, mean_c_norm = calculate_ci_normal(correct)
            axes[i].plot(x_c, y_c, color='skyblue', lw=2)
            axes[i].axvline(mean_c, color='blue', linestyle='--', lw=1, label=r'$\mu$' +f' = {mean_c:.2f}')
            print(f"Correct --> {cohort}:")
            print(f"  Bootstrap: Mean = {mean_c:.2f}, 95% CI = [{ci_lower_c:.2f}, {ci_upper_c:.2f}]")
            print(f"  Normal:    Mean = {mean_c_norm:.2f}, 95% CI = [{ci_lower_c_norm:.2f}, {ci_upper_c_norm:.2f}]")

        if not incorrect.empty:
            # Distribution for Incorrect predictions (KDE)
            kde_i = KDEUnivariate(incorrect)
            kde_i.fit(bw="scott")
            x_i = np.linspace(incorrect.min() - 5, incorrect.max() + 5, 200)
            y_i = kde_i.evaluate(x_i)
            ci_lower_i, ci_upper_i, mean_i = calculate_ci_bootstrap(incorrect)
            ci_lower_i_norm, ci_upper_i_norm, mean_i_norm = calculate_ci_normal(incorrect)
            axes[i].plot(x_i, y_i, color='lightcoral', lw=2)
            axes[i].axvline(mean_i, color='red', linestyle='--', lw=1, label=r'$\mu$' +f' = {mean_i:.2f}')
            print(f"Incorrect --> {cohort}:")
            print(f"  Bootstrap: Mean = {mean_i:.2f}, 95% CI = [{ci_lower_i:.2f}, {ci_upper_i:.2f}]")
            print(f"  Normal:    Mean = {mean_i_norm:.2f}, 95% CI = [{ci_lower_i_norm:.2f}, {ci_upper_i_norm:.2f}]")

        axes[i].set_title(cohort)
        axes[i].set_xlabel(f'Age ({cohort})')
        axes[i].grid(True, linestyle='--', alpha=0.7)
        axes[i].legend()
        print()  # Blank line between cohorts

    axes[0].set_ylabel('Density')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    return fig, axes

# =============================================================================
# POWER ANALYSIS FUNCTIONS
# =============================================================================

def shapiro_power_simulation(n, mean, n_simulations, alpha, shape_param):
    """
    Perform power analysis simulation for Shapiro-Wilk test using gamma distribution.
    Original function from test_power_shapiro.py
    
    Args:
        n: Sample size
        mean: Mean of the distribution
        n_simulations: Number of Monte Carlo simulations
        alpha: Significance level
        shape_param: Shape parameter for gamma distribution
        
    Returns:
        float: Statistical power (proportion of rejections)
    """
    rejections = 0
    # Gamma distribution: mean = k * theta, scale = theta = mean / k
    k = shape_param  # Shape parameter (adjusts skewness)
    scale = mean / k  # Scale parameter to achieve desired mean
    for _ in range(n_simulations):
        # Generate gamma distributed sample
        sample = stats.gamma.rvs(a=k, scale=scale, size=n)
        # Apply Shapiro-Wilk test
        _, p_value = stats.shapiro(sample)
        # Count rejection of H0 (p < alpha)
        if p_value < alpha:
            rejections += 1
    # Calculate power as proportion of rejections
    power = rejections / n_simulations
    return power

# =============================================================================
# MANN-WHITNEY POWER ANALYSIS FUNCTIONS
# =============================================================================

def power_empirical_mw(correct, incorrect, num_sim, shift, alpha=0.05):
    """
    Resamples with replacement and calculates the proportion of p<alpha.
    Original function from test_power_mannwhitney.py
    
    Args:
        correct: Array of correct predictions
        incorrect: Array of incorrect predictions
        alpha: Significance level (default: 0.05)
        num_sim: Number of simulations (default: 5000)
        shift: Shift parameter (default: 0.2)
        
    Returns:
        float: Power (proportion of rejections)
    """
    np.random.seed(42)
    n1, n2 = len(correct), len(incorrect)
    rejections = 0
    for _ in range(num_sim):
        g1 = np.random.choice(correct, size=n1, replace=True)
        g2 = np.random.choice(incorrect, size=n2, replace=True) + shift
        _, p = mannwhitneyu(g1, g2, alternative='two-sided')
        if p < alpha:
            rejections += 1
    return rejections / num_sim

# =============================================================================
# CHI-SQUARED POWER ANALYSIS FUNCTIONS
# =============================================================================

def generate_h1_probs(w, base_probs):
    """
    Generate probabilities for a 2x2 contingency table under H1 with Cohen's w.
    Original function from test_power_chi copy.py
    
    Args:
        w: Cohen's w effect size
        base_probs: Probabilities under H0 from observed data
        
    Returns:
        array: Modified probabilities under H1
    """
    delta = w / np.sqrt(2)  # Simplified perturbation for small w
    h1_probs = base_probs.copy()
    h1_probs[0] += delta  # Male, Correct
    h1_probs[3] += delta  # Female, Incorrect
    h1_probs[1] -= delta  # Male, Incorrect
    h1_probs[2] -= delta  # Female, Correct
    # Ensure valid probabilities
    h1_probs = np.clip(h1_probs, 0, 1)
    h1_probs /= h1_probs.sum()  # Normalize to sum to 1
    return h1_probs

def calculate_power_monte_carlo(n, w, num_sim=10000, base_probs=None, alpha=0.05):
    """
    Calculate statistical power using Monte Carlo simulation.
    Original function from test_power_chi copy.py
    
    Args:
        n: Sample size
        w: Cohen's w effect size
        num_sim: Number of simulations (default: 10000)
        base_probs: Base probabilities from observed data
        alpha: Significance level (default: 0.05)
        
    Returns:
        float: Statistical power (proportion of rejections)
    """
    table_shape = (2, 2)  # 2x2 table
    rejections = 0
    h1_probs = generate_h1_probs(w, base_probs)
    
    for _ in range(num_sim):
        # Simulate contingency table under H1
        counts = np.random.multinomial(n, h1_probs)
        contingency_table = counts.reshape(table_shape)
        
        # Perform Chi-squared test
        try:
            _, p_value, _, _ = chi2_contingency(contingency_table, correction=False)
            
            # Count rejections (power calculation)
            if p_value < alpha:
                rejections += 1
                
        except ValueError:
            # Skip if test fails (e.g., zero expected frequencies)
            continue
    
    # Calculate power as proportion of rejections
    power = rejections / num_sim
    return power