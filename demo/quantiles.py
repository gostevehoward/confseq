import math

from confseq import boundaries
from confseq import quantiles

def quantile_confidence_bound(values, p, lower_radius, upper_radius):
    sorted_values = sorted(values)
    t = len(values)
    lower_index = math.floor(t * (p - lower_radius) + 1)
    lower_confidence_bound = (
        -math.inf if lower_index < 1 else sorted_values[lower_index - 1])
    upper_index = math.ceil(t * (p + upper_radius))
    upper_confidence_bound = (
        math.inf if upper_index > t else sorted_values[upper_index - 1])
    return (lower_confidence_bound, upper_confidence_bound)

def fixed_quantile_confidence_bounds(values, p, t_opt, alpha=0.05,
                                     alpha_opt=0.05):
    t = len(values)
    lower_boundary = boundaries.beta_binomial_mixture_bound(
        p * (1 - p) * t, alpha, v_opt = p * (1 - p) * t_opt, g = 1 - p, h = p,
        alpha_opt=alpha_opt, is_one_sided=False)
    upper_boundary = boundaries.beta_binomial_mixture_bound(
        p * (1 - p) * t, alpha, v_opt=p * (1 - p) * t_opt, g=p, h=1 - p,
        alpha_opt=alpha_opt, is_one_sided=False)
    return quantile_confidence_bound(values, p, lower_boundary / t,
                                     upper_boundary / t)

def uniform_quantile_confidence_bounds(values, p, t_min, alpha=0.05,
                                       use_double_stitching=False):
    sorted_values = sorted(values)
    t = len(values)
    if use_double_stitching:
        upper_bound = quantiles.double_stitching_bound(p, t, alpha, t_min) / t
        lower_bound = (
            quantiles.double_stitching_bound(1 - p, t, alpha, t_min) / t)
    else:
        upper_bound = quantiles.empirical_process_lil_bound(t, alpha, t_min)
        lower_bound = upper_bound
    return quantile_confidence_bound(values, p, lower_bound, upper_bound)
