import math
import jax.numpy as jnp
from jax import random, vmap
from numpyro.distributions import Normal
from jaxtyping import Float, Array

def mean_interval_score(y_true, lq, uq, alpha:float=0.05):
    '''
    Returns the negatively oriented mean interval score for a given
    prediction interval [lq, uq] with confidence level 1 - alpha on 
    a given test set y_true.
    '''
    width = uq - lq
    lower_viol = 2 / alpha * jnp.maximum(0, lq - y_true)
    upper_viol = 2 / alpha * jnp.maximum(0, y_true - uq)

    return jnp.mean(width + lower_viol + upper_viol, axis=-1)

def coverage(y_true, lq, uq):
    '''
    Returns the mean coverage of a given prediction interval [lq, uq] on a
    given test set y_true.
    '''
    return jnp.mean((y_true >= lq) * (y_true <= uq), axis=-1)

def neg_crps_gaussian(y_true, mean, std): 
    '''
    Returns the negatively oriented continuous ranked probability score (CRPS)
    for a given Gaussian predictive distribution with mean and standard deviation
    on a given test set y_true.
    '''
    z = (y_true - mean) / std
    dist = Normal(mean, std)
    
    term1 = 1 / math.sqrt(math.pi)
    term2 = 2 * dist.log_prob(y_true).exp()
    term3 = z * (2 * dist.cdf(z) - 1)

    return jnp.mean(std * (term2 + term3 - term1), axis=-1)

def evaulate_metrics_ensemble(
    y_true: Float[Array, "dim1 1"], 
    y_means: Float[Array, "dim2 dim1"], 
    y_stds: Float[Array, "dim2 dim1"],
    alpha=0.05
):
    '''
    Computes prediction accuracy, mean interval score, coverage probability,
    and CRPS for a given ensemble of predictive Normal distributions with
    means y_means and standard deviations y_stds on a given test set y_true.

    Note that the ensemble means and standard deviations are assumed to have
    dimensions (num_samples, num_observations), where num_samples is the number
    of samples from the posterior predictive distribution and num_observations
    is the number of observations in the test set. These dimensions are expected
    from the output of the Predictive class in NumPyro.

    Args:
        y_true: Test set.
        y_means: Ensemble means.
        y_stds: Ensemble standard deviations.
        alpha: Confidence level for prediction intervals.
    '''
    # Relative root mean squared error of ensemble mean
    rrmse = jnp.sqrt(jnp.mean((y_true - jnp.mean(y_means, axis=-2))**2, axis=-1)) / jnp.std(y_true, axis=-1, ddof=0)

    means = y_means.T
    stds = y_stds.T

    # number of samples-1000
    n_obs = len(y_true)
    num_samples = 1000

    key = random.PRNGKey(0)
    keys = random.split(key, n_obs)

    def single_observation(key, y_true_i, means_i, stds_i):
        key, subkey = random.split(key)
        I = random.randint(subkey, (num_samples,), 0, means_i.shape[0])
        samples = means_i[I] + stds_i[I] * random.normal(key, (num_samples,))

        lq = jnp.quantile(samples, alpha / 2)
        uq = jnp.quantile(samples, 1 - alpha / 2)

        # coverage
        coverage = jnp.mean((y_true_i >= lq) & (y_true_i <= uq))

        # interval score
        width = uq - lq
        lower_viol = 2 / alpha * jnp.maximum(lq - y_true_i, 0)
        upper_viol = 2 / alpha * jnp.maximum(-uq + y_true_i, 0)
        interval_score = lower_viol + upper_viol + width

        # crps
        term1 = jnp.mean(jnp.abs(samples - y_true_i))
        forecasts_diff = jnp.abs(samples[:, None] - samples[None, :])
        term2 = 0.5 * jnp.mean(forecasts_diff)
        crps = term1 - term2

        return coverage, interval_score, crps
    
    coverage, interval_score, crps = vmap(single_observation)(keys, y_true, means, stds)

    return {
        'rrmse': rrmse.item(),
        'mean_is': jnp.mean(interval_score).item(),
        'coverage_prob': jnp.mean(coverage).item(),
        'crps': jnp.mean(crps).item()
    }
