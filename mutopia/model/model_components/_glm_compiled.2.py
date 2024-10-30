
import numpy as np
from numba import njit
from functools import partial
from scipy.sparse.linalg import lsqr
from typing import Callable
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import issparse
from .base import jitpartial, sp2tup, \
    weighted_csr_matmul, transpose_weighted_csr_matmul, \
    csr_matmul
from sklearn.linear_model import ElasticNet


POISSON_GLM = {
    'mean_fn' : njit(lambda eta : np.exp(eta)),
    'response_fn' : njit(lambda y, eta, mu : eta + (y - mu) / mu),
    'weight_fn' : njit(lambda mu : mu),
    'likelihood_fn' : njit(lambda y, w, mu : w @ (y*np.log(mu) - mu)),
}


def interior_solver(f_X, z, w, beta):
    '''
    f_X : callable, like: `lambda beta : X @ beta`
    '''
    pass


def get_sklearn_solver(X, model):
    fit = partial(model.fit, X)

    def _get_sklearn_solver(z, w, beta):
        return fit(z, sample_weight=w).coef_

    return _get_sklearn_solver    


##
# Elastic net interior solver
##
def get_eln_solver(
    X,
    alpha=1e-3,
    l1_ratio=0.95,
    tol=1e-6,
    random_state=None,
): # f(X) -> f(z, w, beta) -> beta
    return get_sklearn_solver(
        X,
        ElasticNet(
            fit_intercept=False,
            warm_start=True,
            alpha=alpha,
            l1_ratio=l1_ratio,
            tol=tol,
            selection='random',
            random_state=random_state,
        )
    )


##
# Ridge regression interior solver
##
def get_CSR_linop(X): # f(X) -> f(w) -> LinearOperator(b) -> yhat
    '''
    Precomputes matrix and matrix-transpose
    '''
    X.eliminate_zeros()
    X = X.tocsr()
    X_T = X.T.tocsr()

    def _get_CSR_linop(weights):
        return LinearOperator(
            dtype=X.dtype,
            shape=X.shape,
            matvec = jitpartial(weighted_csr_matmul, sp2tup(X), weights),
            rmatvec = jitpartial(transpose_weighted_csr_matmul, sp2tup(X_T), weights),
        )
    
    return _get_CSR_linop


def get_lsqr_solver(
    X, 
    alpha=5e-5,
    tol=1e-6
    ): # f(X) -> f(z, w, beta) -> beta
    
    get_linop_fn = get_CSR_linop(X)
    solve = partial(lsqr, atol=tol, btol=tol, damp=np.sqrt(alpha))

    def interior_solver(z, w, beta):
        return solve(get_linop_fn(w), z*w, x0=beta)[0]
    
    return interior_solver


##
# Mixed solver - some weights are regularized, some are not
##
def get_mixed_solver(
        X, 
        is_regularized,
    ): # -> f(X) -> f(solver, solver) -> f(z, w, beta) -> beta

    if not issparse(X):
        raise ValueError('X must be a sparse matrix')

    reg = is_regularized
    X_csc = X.tocsc()
    X_reg = X_csc[:,reg].tocsr()
    X_unreg = X_csc[:, ~reg].tocsr()
    del X_csc

    f_X_reg = jitpartial(csr_matmul, sp2tup(X_reg))
    f_X_unreg = jitpartial(csr_matmul, sp2tup(X_unreg))

    def get_interior(reg_solver, unreg_solver):

        reg_solver = reg_solver(X_reg)
        unreg_solver = unreg_solver(X_unreg)

        def interior_solver(z, w, beta):
            
            beta_new = beta.copy()
            beta_new[~reg] = unreg_solver(z - f_X_reg(beta_new[reg]), w, beta_new[~reg])
            beta_new[reg] = reg_solver(z - f_X_unreg(beta_new[~reg]), w, beta_new[reg])

            return beta_new
        
        return interior_solver
    

    return get_interior



@njit
def outer_update(
    X,
    y, 
    weights, 
    beta,
    *,
    mean_fn,
    response_fn,
    weight_fn,
    likelihood_fn,
):
    eta = csr_matmul(X, beta)
    mu = mean_fn(eta)
    w = weight_fn(mu)
    z = response_fn(y, eta, mu)
    
    return (mu, z, w*weights)


def iter_fit(
    beta,
    tol=1e-3, 
    max_iter=50,
    *,
    outer_update,
    interior_solver,
    likelihood_fn,
    ):

    ll=[]
    for _ in range(max_iter):
        
        mu, z, w = outer_update(beta)
        beta_new = interior_solver(z, w, beta)
        if np.linalg.norm(beta_new - beta) < tol:
            break
        
        beta = beta_new
        ll.append(likelihood_fn(mu))
        print(ll[-1])

    return beta #, ll


def make_optimizer(
    X,
    tol=5e-4,
    max_iter=100
): # f(X) -> f(solver, y, weights) -> f(beta) -> beta
    
    if not issparse(X):
        raise ValueError('X must be a sparse matrix')

    update_fn = partial(
        outer_update,
        sp2tup(X),
        **POISSON_GLM,
    )

    def optim_fn(solver, y, weights):
        return partial(
            iter_fit,
            tol=tol,
            max_iter=max_iter,
            outer_update=partial(update_fn, y, weights),
            interior_solver=solver,
            likelihood_fn=partial(POISSON_GLM['likelihood_fn'], y, weights),
        )
    
    return optim_fn
