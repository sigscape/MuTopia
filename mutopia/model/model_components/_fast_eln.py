import numpy as np
from sklearn.linear_model import _cd_fast as cd_fast
from sklearn.utils import check_array, check_random_state
from sklearn.exceptions import ConvergenceWarning
import warnings

def get_eln_solver(
    X,
    *,
    alpha=1e-3,
    l1_ratio=0.5,
    random_state=None,
    tol=1e-4,
    max_iter=1000,
    selection="cyclic", # use cyclic to make it deterministic
):

    X = check_array(
        X,
        accept_sparse="csc",
        dtype=[np.float64, np.float32],
        order="F",
        copy=True,
    )

    def solver(y, sample_weight, coef_init):

        y = np.asfortranarray(
            y,
            dtype = X.dtype,
        )

        # renormalize and make F-ordered
        sample_weight = np.asfortranarray(
            sample_weight/np.sum(sample_weight) * X.shape[0],
            dtype=X.dtype,
        )

        coef_ = np.asfortranarray(coef_init, dtype=X.dtype)
            
        n_samples, n_features = X.shape
        X_sparse_scaling = np.zeros(n_features, dtype=X.dtype)
        
        if selection not in ["random", "cyclic"]:
            raise ValueError("selection should be either random or cyclic.")
        random = selection == "random"

        l1_reg = alpha * l1_ratio * n_samples
        l2_reg = alpha * (1.0 - l1_ratio) * n_samples

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")#, category=ConvergenceWarning)

            model = cd_fast.sparse_enet_coordinate_descent(
                w=coef_,
                alpha=l1_reg,
                beta=l2_reg,
                X_data=X.data,
                X_indices=X.indices,
                X_indptr=X.indptr,
                y=y,
                sample_weight=sample_weight,
                X_mean=X_sparse_scaling,
                max_iter=max_iter,
                tol=tol,
                rng=random_state,
                random=random,
                positive=False,
            )

        coef_, dual_gap_, eps_, n_iter_ = model

        return coef_

    return solver