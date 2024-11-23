
import numpy as np
from numba import njit
from functools import partial
from scipy.sparse.linalg import lsqr
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import issparse
from .base import sp2tup, weighted_csr_matmul, \
    transpose_weighted_csr_matmul, csr_matmul, \
    design_csr
from sklearn.linear_model import ElasticNet
from sklearn import set_config
set_config(skip_parameter_validation=True)


POISSON_GLM = {
    'mean_fn' : njit(lambda eta : np.exp(eta), nogil=True),
    'response_fn' : njit(lambda y, eta, mu : eta + (y - mu) / mu, nogil=True),
    'weight_fn' : njit(lambda mu : mu, nogil=True),
    'likelihood_fn' : njit(lambda y, w, mu : w @ (y*np.log(mu) - mu), nogil=True),
}


def get_sklearn_solver(X, model):
    fit = partial(model.fit, X)

    def _get_sklearn_solver(z, w, beta):
        return fit(z, sample_weight=w).coef_

    return _get_sklearn_solver    

##
# Ridge regression interior solver
##
def get_CSR_linop(
    X,
    matvec = weighted_csr_matmul,
    rmatvec = transpose_weighted_csr_matmul,
): # f(X) -> f(w) -> LinearOperator(b) -> yhat
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
            matvec = partial(matvec, sp2tup(X), weights),
            rmatvec = partial(rmatvec, sp2tup(X_T), weights),
        )
    
    return _get_CSR_linop



def get_lsqr_solver(
    X, 
    alpha=5e-5,
    tol=1e-6
    ): # f(X, alpha, tol) -> f(z, w, beta) -> beta
    
    get_linop_fn = get_CSR_linop(X)
    solve = partial(lsqr, atol=tol, btol=tol, damp=np.sqrt(alpha))

    def interior_solver(z, w, beta):
        return solve(get_linop_fn(w), z*w, x0=beta*0.97)[0]
    
    return interior_solver



@njit(nogil=True)
def _partial_ls_update(
    XT,
    z, w, beta,
    alpha=1e-4,
):
    X_denom = design_csr(XT, w) + np.sum(w)*alpha
    return design_csr(XT, z*w)/X_denom


def partial_ls_solver(
    X,
    alpha=1e-4,
):
    X.eliminate_zeros()
    X = X.tocsc()
    return partial(
        _partial_ls_update,
        sp2tup(X.T.tocsr()),
        alpha=alpha,
    )


@njit(nogil=True)
def _iterative_partial_ls(
        X1, X1T,
        X2, X2T,
        group_mask,
        z, w, beta,
        tol=1e-4, 
        max_iter=10,
        alpha=1e-4,
    ):

    X2_denom = design_csr(X2T, w) + np.sum(w)*alpha
    X1_denom = design_csr(X1T, w) + np.sum(w)*alpha
    
    for i in range(max_iter):
    
        beta_old = beta.copy()
        r = z - design_csr(X1, beta[group_mask])
        beta[~group_mask] = design_csr(X2T, r*w)/X2_denom

        r = z - design_csr(X2, beta[~group_mask])
        beta[group_mask] = design_csr(X1T, r*w)/X1_denom

        if np.linalg.norm(beta-beta_old) < tol:
            break

    return beta


def interative_partial_ls_solver(
    X,
    max_iter=10,
    tol=1e-4,
    alpha=1e-4,
    *,
    group_mask, # at most two groups okay...
):
    '''
    Partial derivative solver for the least squares problem.
    If the feature matrix is separable into two (or more, but this is not implemented) groups
    such that within that group, each feature behaves as an intercept, then we can solve the
    optimal coefficient in one step.

    This solver iterates between updating the coefficients of two groups of features
    until convergence, and does this without QR decomposition or any matrix inversion.
    '''
    
    def tup_X_XT(X):
        return  sp2tup(X.tocsr()), sp2tup(X.T.tocsr())

    X.eliminate_zeros()
    X = X.tocsc()

    return partial(
        _iterative_partial_ls,
        *tup_X_XT(X[:, group_mask]),
        *tup_X_XT(X[:, ~group_mask]),
        group_mask,
        max_iter=max_iter,
        alpha=alpha,
        tol=tol,
    )


def right_intercept_solver(
    X,
    *,
    solver,
):
    
    X = X.tocsc()[:,:-1].tocsr()  
    f_X = partial(csr_matmul, sp2tup(X))
    solver = solver(X)

    def interior_solver(z, w, beta):
        new_beta = beta.copy()
        intercept = w/np.sum(w) @ (z - f_X(new_beta[:-1]))
        
        new_beta[-1] = intercept
        new_beta[:-1] = solver(z - intercept, w, new_beta[:-1])

        return new_beta
    
    return interior_solver


##
# Mixed solver - some weights are regularized, some are not
##
def setup_mixed_solver(
        X,
        *,
        is_regularized,
        reg_solver,
        unreg_solver,
    ): # -> f(X) -> f(solver, solver) -> f(z, w, beta) -> beta
    '''
    Sets up a solver where some of the coefficients are optimized
    by one model, and the rest are optimized by another model.
    '''

    if not issparse(X):
        raise ValueError('X must be a sparse matrix')

    reg = is_regularized
    X_csc = X.tocsc()
    X_reg = X_csc[:,reg].tocsr()
    X_unreg = X_csc[:, ~reg].tocsr()
    del X_csc

    f_X_reg = partial(csr_matmul, sp2tup(X_reg))
    f_X_unreg = partial(csr_matmul, sp2tup(X_unreg))

    reg_solver = reg_solver(X_reg)
    unreg_solver = unreg_solver(X_unreg)

    def interior_solver(z, w, beta):
        
        beta_new = beta.copy()
        beta_new[~reg] = unreg_solver(z - f_X_reg(beta_new[reg]), w, beta_new[~reg])
        beta_new[reg] = reg_solver(z - f_X_unreg(beta_new[~reg]), w, beta_new[reg])

        return beta_new
        
    return interior_solver



@njit(nogil=True)
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

    #ll=[]
    for _ in range(max_iter):
        
        _, z, w = outer_update(beta)
        beta_new = interior_solver(z, w, beta)

        if np.linalg.norm(beta_new - beta) < tol:
            break
        
        beta = beta_new
        #ll.append(likelihood_fn(mu))

    return beta #, ll



def _optim_fn(solver, update_fn, y, weights, tol=1e-3, max_iter=50):
    return partial(
        iter_fit,
        tol=tol,
        max_iter=max_iter,
        outer_update=partial(update_fn, y, weights), # f(beta) -> mu, z, w
        interior_solver=solver, # f(z, w, beta) -> beta
        likelihood_fn=None 
    )


def make_optimizer(
    X,
    solver,
    tol=5e-4,
    max_iter=100,
):
    
    update_fn = partial(
        outer_update,
        sp2tup(X),
        **POISSON_GLM,
    )

    return partial(
        _optim_fn,
        solver(X),
        update_fn,
        tol=tol,
        max_iter=max_iter,
    )