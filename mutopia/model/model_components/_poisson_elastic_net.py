from sklearn.linear_model import LinearRegression, ElasticNet
from scipy import sparse
import numpy as np
from functools import partial
from sklearn.linear_model._base import _rescale_data
from scipy.sparse import linalg as sp_linalg


def simple_ridge(
    X,
    y,
    sample_weight,
    beta0,
    *,
    alpha,
    max_iter,
    tol,
):
    """
    The scikit-learn Ridge class essentially does the following two steps.
    I've just chopped out the bloat code, and also added propogation of the
    initial guess beta0.
    """

    X, y, _ = _rescale_data(
        X.copy(),
        y.copy(),
        sample_weight.copy(),
    )

    coefs = sp_linalg.lsqr(
        X,
        y,
        damp=np.sqrt(alpha),
        atol=tol,
        btol=tol,
        iter_lim=max_iter,
        x0=beta0 * 0.97,  # apply a tiny bit of shrinkage to the initial guess
    )[0]

    return coefs


POISSON_GLM = {
    "eta_fn": lambda X, B: X @ B,
    "mean_fn": lambda eta: np.exp(eta),
    "response_fn": lambda y, eta, mu: eta + (y - mu) / mu,
    "weight_fn": lambda mu: mu,
    "likelihood_fn": lambda y, w, mu: w @ (y * np.log(mu) - mu),
}


class SklearnWrapper:

    def __init__(self, model):
        self.model = model
        self._n_calls = 0

    def __call__(self, X, z, w, _):
        self._n_calls += 1
        return self.model.fit(
            X,
            z,
            sample_weight=w,
            # check_input=self._n_calls==1,
        ).coef_


def ols_interior():
    return SklearnWrapper(LinearRegression(fit_intercept=False))


def elastic_net_interior(alpha=0.1, l1_ratio=0.99):
    model = ElasticNet(
        fit_intercept=False, warm_start=True, alpha=alpha, l1_ratio=l1_ratio
    )
    return SklearnWrapper(model)


def mixed_regularization_interior(
    *,
    X,
    regularized_model,
    unregularized_model,
    regularize_mask,
):

    m = regularize_mask.copy()
    do_reg = any(m)

    if sparse.issparse(X):
        X = X.tocsc()
        X_reg = X[:, m].tocsr()
        X_unreg = X[:, ~m].tocsr()
    else:
        X_reg = X[:, m]
        X_unreg = X[:, ~m]

    def beta_update(_, z, w, beta):

        beta_new = beta.copy()

        beta_new[~m] = unregularized_model(
            X_unreg,
            z - X_reg @ beta_new[m],
            w,
            beta[~m],
        )

        if do_reg:
            beta_new[m] = regularized_model(
                X_reg,
                z - X_unreg @ beta_new[~m],
                w,
                beta[m],
            )

        return beta_new

    return beta_update


def get_irls_beta_update(
    X,
    y,
    weights,
    *,
    eta_fn,
    mean_fn,
    response_fn,
    weight_fn,
    likelihood_fn,
    interior_solver,
):
    """
    Performs iteratively reweighted least squares for a generalized linear model.

    Parameters
    ----------
    eta_fn : F(X,B) -> [float] (eta = X @ B for linear models)
    likelihood_fn : F(y, weight, mu) -> [float]
        The likelihood function.
    mean_fn : F(eta) -> [mu]
    response_fn : F(y, eta, mu) -> [float]
    weight_fn : F(mu) -> [float]
    interior_solver : F(X, z, w) -> B
        The interior solver to solve the weighted least squares problem.
    """
    eta_fn = partial(eta_fn, X)
    response_fn = partial(response_fn, y)
    likelihood_fn = partial(likelihood_fn, y, weights)
    interior_solver = partial(interior_solver, X)

    def update_fn(beta):
        try:
            eta = eta_fn(beta)
            mu = mean_fn(eta)
            w = weight_fn(mu)
            z = response_fn(eta, mu)
        except FloatingPointError:
            print(beta, y)

        return interior_solver(z, w * weights, beta)

    return update_fn, lambda beta: likelihood_fn(mean_fn(eta_fn(beta)))


def iter_fit(update_fn, ll_fn, beta0, tol=1e-3, max_iter=50):

    beta = beta0.copy()
    ll = []

    for _ in range(max_iter):

        beta_new = update_fn(beta)
        ll.append(ll_fn(beta_new))

        if np.linalg.norm(beta_new - beta) < tol:
            break

        beta = beta_new

    return beta


"""def optim_coefs_noreg(
        beta0,
        X,
        y,
        sample_weights,
        tol=5e-4,
        max_iter=100,
        **kwargs
):
    return PoissonRegressor(
        fit_intercept=False,
        alpha=5e-4,
        tol=tol,
        warm_start=True,
        solver='lbfgs',        
        ).fit(X, y, sample_weight=sample_weights).coef_"""


def optim_coefs_noreg(beta0, X, y, sample_weights, tol=5e-4, max_iter=100, **kwargs):
    update_fn, ll_fn = get_irls_beta_update(
        X=X, y=y, weights=sample_weights, **POISSON_GLM, interior_solver=ols_interior()
    )

    return iter_fit(update_fn, ll_fn, beta0, tol=tol, max_iter=max_iter)


def make_optimizer(
    X,
    tol=5e-4,
    max_iter=100,
    *,
    regularized_model,
    unregularized_model,
    regularize_mask,
):

    solver = mixed_regularization_interior(
        X=X,
        regularized_model=regularized_model,
        unregularized_model=unregularized_model,
        regularize_mask=regularize_mask,
    )

    def optim_inner(
        beta0,
        y,
        sample_weights,
    ):
        update_fn, ll_fn = get_irls_beta_update(
            X=X, y=y, weights=sample_weights, **POISSON_GLM, interior_solver=solver
        )

        return iter_fit(update_fn, ll_fn, beta0, tol=tol, max_iter=max_iter)

    return optim_inner


"""
def stagewise_optim_coefs(
        X,
        y,
        sample_weights,
        *,
        regularized_model, 
        regularize_mask,
):
    
    m = regularize_mask.copy()
    do_reg = any(m)

    if sparse.issparse(X):
        X = X.tocsc()
        X_reg = X[:,m].tocsr()
        X_unreg = X[:,~m].tocsr()
    else:
        X_reg = X[:,m]
        X_unreg = X[:,~m]

    basemodel=PoissonRegressor(
        fit_intercept=False, 
        alpha=1e-5, 
        tol=1e-3,
        warm_start=True,
        solver='lbfgs',
    )

    get_interior_solver = partial(
        get_irls_beta_update,
        X=X_reg,
        **POISSON_GLM,
        interior_solver=delegate_interior_update(regularized_model)
    )

    def reweight_obs(y,y_hat,w):
        return y/y_hat, w * y_hat

    def beta_update(beta):
        
        beta_new = beta.copy()
        
        beta_new[~m] = basemodel \
            .fit( X_unreg, *reweight_obs(y, np.exp(X_reg  @ beta_new[m]), sample_weights) )\
            .coef_

        if do_reg:
            _y, _weights = reweight_obs(y, np.exp(X_unreg  @ beta_new[~m]), sample_weights)
            _update_fn, _ = get_interior_solver(y=_y, weights=_weights,)
            beta_new[m] = _update_fn(beta_new[m])
        
        return beta_new
    
    def ll_fn(beta):
        return POISSON_GLM['likelihood_fn'](y, sample_weights, np.exp(X @ beta))

    return lambda beta0 : iter_fit(beta_update, ll_fn, beta0)"""
