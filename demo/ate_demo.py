import collections
import math
import os

import numpy as np
import pandas as pd

from confseq import boundaries

def form_predictions(outcomes, selector, default_prediction):
    means = np.cumsum(outcomes * selector) / np.maximum(1, np.cumsum(selector))
    lagged_means = np.roll(means, 1)
    lagged_means[0] = default_prediction
    return lagged_means

def estimate_average_treatment_effect(outcomes, treatments, propensity,
                                      support_bounds, expected_outcome_noise,
                                      optimized_t, y1_predictions=None,
                                      y0_predictions=None, coverage_alpha=0.05,
                                      alpha_opt=0.05):
    support_center = (support_bounds[0] + support_bounds[1]) / 2.0
    v_opt = (optimized_t * expected_outcome_noise
             * (1 / propensity + 1 / (1 - propensity)))
    if y1_predictions is None:
        y1_predictions = form_predictions(outcomes, treatments == 1,
                                          support_center)
    if y0_predictions is None:
        y0_predictions = form_predictions(outcomes, treatments == 0,
                                          support_center)

    tau_hat = y1_predictions - y0_predictions
    weights = (treatments - propensity) / (propensity * (1 - propensity))
    predictions = np.where(treatments == 1, y1_predictions, y0_predictions)
    Xt = tau_hat + weights * (outcomes - predictions)
    St = np.cumsum(Xt)
    Vt = np.cumsum((Xt - tau_hat)**2)
    p_min = min(propensity, 1 - propensity)
    support_diameter = support_bounds[1] - support_bounds[0]
    c = 2 * support_diameter / p_min
    t_array = np.arange(1.0, len(outcomes) + 1.0)
    p_value = np.exp(-boundaries.gamma_exponential_log_mixture(
             St, Vt, v_opt, c, alpha_opt=alpha_opt / 2))
    confidence_radius = (
        1.0 / t_array * boundaries.gamma_exponential_mixture_bound(
            Vt, coverage_alpha / 2, v_opt, c, alpha_opt=alpha_opt / 2))
    return pd.DataFrame(collections.OrderedDict([
        ('t', t_array),
        ('point_estimate', St / t_array),
        ('upper_confidence_bound', np.minimum(
            support_diameter, St / t_array + confidence_radius)),
        ('lower_confidence_bound', np.maximum(
            -support_diameter, St / t_array - confidence_radius)),
        ('p_value', np.minimum(p_value, 1.0))]))

def check_approx_equal(name, actual, expected, tolerance=1e-4):
    abs_errors = np.abs(expected - actual)
    any_error = (abs_errors.max() > tolerance)
    if any_error:
        error_indexes = (abs_errors > tolerance).to_numpy().nonzero()[0]
        error_data = pd.DataFrame({
            'index': error_indexes,
            'expected': expected[error_indexes],
            'actual': actual[error_indexes]})
        print('Errors in {}'.format(name))
        print(error_data)
    return any_error

def test_ate():
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'ate_data.csv'))
    results = estimate_average_treatment_effect(
        data['y'], data['treatment'], 0.5, [0, 1], 0.2 * 0.8, 50)
    error_indicators = [
        check_approx_equal('upper confidence bounds',
                           results['upper_confidence_bound'],
                           data['upper_confidence_bound']),
        check_approx_equal('lower confidence bounds',
                           results['lower_confidence_bound'],
                           data['lower_confidence_bound']),
        check_approx_equal('point estimates',
                           results['point_estimate'],
                           data['point_estimate']),
        check_approx_equal('p-values', results['p_value'], data['p_value'])]
    if not any(error_indicators):
        print('Results match expectations')

if __name__ == '__main__':
    test_ate()
