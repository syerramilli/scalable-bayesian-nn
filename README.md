# Scalable Bayesian Neural Networks for regression 

In this project, we test different inference schemes for Bayesian Neural Networks (BNNs) on several regression datasets. The primary interest in this project is to assess the quality of the uncertainty estimates provided by the different inference schemes, while also considering the computational cost of the inference. Additionally, we will also assess the prediction accuracy as we also want accurate predictions from our model - models that produce many inaccurate predictions with large uncertainties are not useful in practice.

We will use primarily use JAX and NumPyro for the implementation of the BNNs and the inference schemes. We might use other JAX based libraries for some inference procedures such as stochastic gradient Langevin dynamics (SGLD) and Stochastic Gradient Hamiltonian Monte Carlo (HMC). We will fetch the datasets from the [OpenML](https://www.openml.org/).

## Previous project
This project has similar aims and scopes as our previous project at https://github.com/syerramilli/iems490bnn/. In that project, we used implementations within the PyTorch ecosystem (such as PyRO for variational inference), while we will focus on inference schemes that have active implementations within the JAX ecosystem. 

From the previous project, we found that variational inference schemes have worse uncertainty estimates than even simple ensemble based neural network models. This is likely because we used Normal distributions for the variational posterior, which has problems in approximating multi-modal posteriors. See [Fort et. al. 2020](https://arxiv.org/abs/1912.02757) for more details. In the current project, we will consider more expressive variational posteriors such as normalizing flows and normalizing flows with autoregressive transforms.

## Metrics for assessing uncertainty quantification

We will use the following metrics to assess the quality of the uncertainty estimates provided by the different inference schemes:

1. Coverage probability for the 95% prediction intervals
2. Negatively oriented mean interval score (IS) for the 95% prediction intervals
3. Negatively oriented continuous ranked probability score (CRPS) for the predictive distributions

The latter two are proper scoring rules defined in a negative orientation. See [Gneiting and Raftery 2007](https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf) for more details on proper scoring rules. The 95% IS measures the quality of a 95% credible interval, while the CRPS measures the quality of the entire predictive distribution.