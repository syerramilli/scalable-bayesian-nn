import jax.numpy as jnp
import numpyro
from numpyro.distributions import Normal, Gamma
from numpyro.contrib.module import random_flax_module
import flax.linen as nn
from typing import Optional

class FlaxMLP(nn.Module):
    n_layers: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        # hidden layers
        for i in range(self.n_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)

        # output layer has no activation
        x = nn.Dense(1)(x)
        return x
    
def bnn_model(
    x, y=None,
    n_layers:int=1,
    hidden_dim:int=25,
    subsample_size:Optional[int] = None
):
    '''
    Bayesian neural network model with priors on weights and biases
    as described in Liu, Q., & Wang, D. (2016). Stein variational
    gradient descent: A general purpose Bayesian inference algorithm.
    Advances in neural information processing systems, 29.

    This uses a Flax module to define the neural network.

    Args:
        x: Input data.
        y: Target data.
        n_layers: Number of hidden layers in the neural network (Default: 1)
        hidden_dim: Number of hidden units in the neural network (Default: 25)
        subsample_size: Size of the subsample used for inference. In each
            optimization step, a subsample of size subsample_size is drawn
            from the training data. If None, the full training data is used.
    '''

    # hyperprior for the precision of the weights and biases
    gamma = numpyro.sample('gamma', Gamma(1.0, 0.1))

    # get dimensions
    n, d = x.shape

    # Neural network
    mlp = random_flax_module(
        name="mlp",
        nn_module=FlaxMLP(n_layers, hidden_dim),
        prior=Normal(0, 1 / jnp.sqrt(gamma)),
        input_shape=(d,)
    )

    # precision parameter for observations
    prec_obs = numpyro.sample('prec_obs', Gamma(1.0, 0.1))

    with numpyro.plate("data", n, subsample_size=subsample_size, dim=-1):
        batch_x = numpyro.subsample(x, event_dim=1)
        if y is not None:
            batch_y = numpyro.subsample(y, event_dim=0)
        else:
            batch_y = y
        
        mean_y = numpyro.deterministic('y_pred', mlp(batch_x).ravel())
        numpyro.sample(
            "y", Normal(mean_y, 1 / jnp.sqrt(prec_obs)).to_event(1), obs=batch_y
        )



def numpyro_bnn_model(
    x, y=None, 
    hidden_dim:int=50,
    subsample_size:int=100
):
    '''
    Single hidden layer Bayesian neural network model with priors
    on weights and biases as described in Liu, Q., & Wang, D. (2016). 
    Stein variational gradient descent: A general purpose Bayesian 
    inference algorithm. Advances in neural information processing 
    systems, 29.

    Args:
        x: Input data.
        y: Target data.
        hidden_dim: Number of hidden units in the neural network.
        subsample_size: Size of the subsample used for inference.
    '''

    # hyperprior for the precision of the weights and biases
    gamma = numpyro.sample('gamma', Gamma(1.0, 0.1))

    # Set up dimensions
    n, d = x.shape

    # hidden layer
    b1 = numpyro.sample('b1', Normal(0, 1 / jnp.sqrt(gamma)).expand([hidden_dim]))
    w1 = numpyro.sample('w1', Normal(0, 1 / jnp.sqrt(gamma)).expand([d, hidden_dim]))

    # output layer
    w2 = numpyro.sample('w2', Normal(0, 1 / jnp.sqrt(gamma)).expand([hidden_dim, 1]))    
    b2 = numpyro.sample('b2', Normal(0, 1 / jnp.sqrt(gamma)).expand([1]))

    # precision parameter for observations
    prec_obs = numpyro.sample('prec_obs', Gamma(1.0, 0.1))

    # observation model
    with numpyro.plate("data", n, subsample_size=subsample_size, dim=-1):
        batch_x = numpyro.subsample(x, event_dim=1)
        if y is not None:
            batch_y = numpyro.subsample(y, event_dim=0)
        else:
            batch_y = y
        
        mean_y = numpyro.deterministic(
            'y_pred', (jnp.maximum(batch_x @ w1 + b1, 0) @ w2 + b2).ravel()
        )

        numpyro.sample(
            "y", Normal(mean_y, 1 / jnp.sqrt(prec_obs)), obs=batch_y
        )