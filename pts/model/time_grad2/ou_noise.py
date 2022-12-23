import numpy as np
import scipy.linalg as la

class OrnsteinUhlenbeckProcess:
    """
    Ornstein-Uhnlenbeck process
    Based on https://github.com/tudortrita/Imperial-College/blob/bd3b2d6e27574afb4c117d142ce007dfa34ba25f/M4A44-Computational-Stochastic-Processes/code/Week%203/w3_gaussian_processes.ipynb
    """

    def __init__(self, shape, mu=0.0, theta=0.05, sigma=1):
        self.shape = shape  # batch and action dimension
        b, T = self.b, self.T = shape
        self.mu = mu
        self.theta = theta
        self.sigma = sigma

        s = np.linspace(0, T, T)
        t = np.linspace(0, T, T)

        # Make correlation matrix
        grid_S, grid_T = np.meshgrid(s, t)
        self.correlation_matrix = self.ou_cov(grid_S, grid_T)

        # Sigma needs to be positive definite to use the Cholesky factorization.
        # We will therefore use the matrix square root.
        # self.C = np.linalg.cholesky(self.correlation_matrix)
        self.C = la.cholesky(self.correlation_matrix, lower=True)

    def ou_cov(self, t, s):
        # https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
        # theta and mu are the coefficients in the drift, sigma is the
        # diffusion, and x0 is the initial condition.
        #the correlation matrix for that stationary process
        return (
            self.sigma**2
            / (2 * self.theta)
            *
            (
                np.exp(-self.theta * np.abs(t - s))
                # - np.exp(-self.theta * np.abs(t + s))
            )
        )

    def sample_decomposed(self):
        # Generate m samples from the standard mutivariate normal distribution,
        # with which we will construct the other processes.
        # (Each line is a sample from the Gaussian)
        x = np.random.randn(self.b, self.T)
        return self.C, x

    def compose(self, C, x):
        # TODO I could think about random mu and theta so that I get differen't freq and starting points
        # otherwise the first step is easy
        return self.mu + C.dot(x.T).T

    def sample(self):
        C, x = self.sample_decomposed()
        return C, x, self.compose(C, x)
    
