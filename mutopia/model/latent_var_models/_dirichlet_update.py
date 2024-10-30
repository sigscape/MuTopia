
import numpy as np
from scipy.special import psi, gammaln, polygamma, xlogy
from scipy.optimize import line_search
import logging
logger = logging.getLogger(' Prior update')


def dirichlet_multinomial_logprob(z, alpha):

    z, num = np.unique(z, return_counts=True)

    n_z = np.zeros_like(alpha)
    n_z[z] = num

    n = sum(n_z)

    alpha_bar = sum(alpha)

    return gammaln(alpha_bar) + gammaln(n+1) - gammaln(n+alpha_bar) + \
            np.sum(
                gammaln(n_z + alpha) - gammaln(alpha) - gammaln(n_z + 1)
            )


def log_dirichlet_expectation(alpha):
    
    if len(alpha.shape) == 1:
        return psi(alpha) - psi(np.sum(alpha))
    else:
        return psi(alpha) - psi(np.sum(alpha, axis = -1, keepdims=True))


def dirichlet_bound(alpha, gamma):

    logE_gamma = log_dirichlet_expectation(gamma)

    alpha = np.expand_dims(alpha, 0)
    
    return gammaln(np.sum(alpha)) - gammaln(np.sum(gamma, axis = -1)) + \
        np.sum(
             gammaln(gamma) - gammaln(alpha) + (alpha - gamma)*logE_gamma,
             axis = -1
        )

class NoImprovementError(ValueError):
    pass

def _dir_prior_objective(alpha, N, logphat):
    return -N * (gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)) + np.sum((alpha - 1) * logphat))


def _dir_prior_update_step(prior, N, logphat):

    def _gradient(alpha):
        return -N * (psi(np.sum(alpha)) - psi(alpha) + logphat)
    
    gradf = -_gradient(prior)

    c = N * polygamma(1, np.sum(prior))
    q = -N * polygamma(1, prior)

    b = np.sum(gradf / q) / (1 / c + np.sum(1 / q))

    dprior = -(gradf - b) / q

    step_size, *search = line_search(
        lambda x : _dir_prior_objective(x, N, logphat),
        _gradient,
        prior,
        dprior,
        amax = 1.,
        maxiter = 100
    )

    if step_size is None:
        if np.linalg.norm(gradf) < 1e-4:
            logger.debug('Prior has converged.')
            step_size = 0
        else:
            logger.debug('Newton step cannot improve log-likelihood of prior.')
            raise NoImprovementError()

    if not step_size == 1:
        logger.debug(f'Line search found a better step size: {step_size}')
    
    new_prior = step_size*dprior + prior

    if np.any(new_prior < 0.01):
        logger.debug('Performing projected gradient descent update.')
        new_prior = np.maximum(new_prior, 0.01)
    
    return new_prior


def _dir_prior_newton_iter(prior, N, logphat, 
                           compare_prior = None, 
                           max_iter = 100):
    
    if compare_prior is None:
        compare_prior = prior

    curr_val = _dir_prior_objective(compare_prior, N, logphat)

    try:
        for iter in range(max_iter):
            old_prior = prior.copy()
            prior = _dir_prior_update_step(old_prior, N, logphat)
            if np.abs(old_prior - prior).sum() < 1e-3:
                break
        iter+=1
    except NoImprovementError:
        pass
    else:
        logger.debug(f'Prior updated in {iter} iterations.')

    new_val = _dir_prior_objective(prior, N, logphat)
    
    if iter == 0 or new_val > curr_val:
        raise NoImprovementError()
    
    logger.debug(f'Prior log-likelihood improvement: {curr_val - new_val}\nNew prior: {" ".join(map(str,prior))}')
    return prior
        

def update_dir_prior(prior, N, logphat):

    old_prior = prior.copy()

    try:
        prior = _dir_prior_newton_iter(prior, N, logphat)
    except NoImprovementError:
        
        try:
            logger.debug('Re-evaluating prior with different initial point.')
            p_tild = np.exp(logphat)*0.01

            prior = _dir_prior_newton_iter(p_tild, N, logphat, compare_prior = prior)

        except NoImprovementError:
            logger.warn('Failed to update prior, reverting to old value.')
            prior = old_prior
    
    return prior


def update_tau(mu, nu):
    return np.sqrt(2*np.pi) * np.sum(mu**2 + nu**2, axis = -1)


def update_alpha(alpha, gamma):
    
    N = gamma.shape[0]
    log_phat = log_dirichlet_expectation(gamma).mean(-2)
    return update_dir_prior(alpha, N, log_phat)