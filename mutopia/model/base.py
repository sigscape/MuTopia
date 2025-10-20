from __future__ import annotations

import numpy as np
import xarray as xr
from functools import partial, wraps
from joblib import dump
from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod

from mutopia.utils import logger, ParContext, parallel_map
from mutopia.gtensor import dims_except_for, train_test_split
from mutopia.gtensor.validation import check_corpus
from mutopia.gtensor.dtypes import get_mode_config

from .model.optim import fit_model as _fit_model_impl
from .model.model_components import *
from .model.latent_var_models import *
from .model.factor_model import FactorModel

# interfaces
from .mixture_model.mixture_interface import MixtureInterface as MIX
from .model.gtensor_interface import GtensorInterface as GTI
from .mixture_model import DenseMixtureModel, SparseMixtureModel, SharedExposuresMixtureModel

# typing helpers
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Prefer forward references at runtime; only import for type checkers
    from mutopia.gtensor.gtensor import GTensorDataset
    from numpy.random import RandomState

"""
The Model class is a wrapper around a trained model state object, 
and provides the high-level interface for interacting with the model.
This is the entry point for the user to interact with and annotate data.
"""

__all__=["TopographyModel"]


class DenseSharedMixtureModel(DenseMixtureModel, SharedExposuresMixtureModel):
    pass


class SparseSharedMixtureModel(SparseMixtureModel, SharedExposuresMixtureModel):
    pass


class TopographyModel(ABC, BaseEstimator):
    """
    The TopographyModel is the heart of the mutopia modeling framework. It decomposes 
    genome topography data into discrete components, simultaneously capturing their
    genomic distributions and spectra. MuTopia models are memory and compute efficient,
    and can be accelerated using multiple-threading. 

    The models mostly work like Scikit-learn estimators. After fitting a model, 
    you can use it to annotate your data using the `annot_data` method.

    Parameters
    ----------
    num_components : int, default=15
        Number of components in the signature model.
    init_components : list, default=[]
        List of initial components to use.
    fix_components : list, default=[]
        List of components to keep fixed during optimization.
    seed : int, default=0
        Random seed for reproducibility.
    context_reg : float, default=0.0001
        Regularization parameter for context model.
    context_conditioning : float, default=1e-9
        Conditioning parameter for context model.
    conditioning_alpha : float, default=1e-9
        Alpha parameter for conditioning.
    pi_prior : float, default=1.0
        Prior parameter for pi in the locals model.
    tau_prior : float, default=1.0
        Prior parameter for tau in the locals model.
    locus_model_type : str, default="gbt"
        Type of model for locus. Gradient Boosted Trees by default.
    tree_learning_rate : float, default=0.15
        Learning rate for tree-based models.
    max_depth : int, default=5
        Maximum depth of trees in the locus model.
    max_trees_per_iter : int, default=25
        Maximum number of trees per iteration.
    max_leaf_nodes : int, default=31
        Maximum number of leaf nodes in each tree.
    min_samples_leaf : int, default=30
        Minimum number of samples required at a leaf node.
    max_features : float, default=1.0
        Fraction of features to consider when looking for best split.
    n_iter_no_change : int, default=1
        Number of iterations with no improvement to wait before early stopping.
    use_groups : bool, default=True
        Whether to use groups in the model.
    add_corpus_intercepts : bool, default=False
        Whether to add corpus-specific intercepts.
    convolution_width : int, default=0
        Width of convolution window.
    l2_regularization : float, default=1
        L2 regularization strength.
    max_iter : int, default=25
        Maximum number of iterations for the optimization.
    init_variance_theta : float, default=0.03
        Initial variance for theta parameters.
    init_variance_context : float, default=0.1
        Initial variance for context parameters.
    empirical_bayes : bool, default=True
        Whether to use empirical Bayes for parameter estimation.
    begin_prior_updates : int, default=50
        Iteration to begin prior updates.
    stop_condition : int, default=50
        Stopping condition for optimization.
    num_epochs : int, default=2000
        Number of epochs for training.
    locus_subsample : float or None, default=None
        Fraction of loci to subsample in each iteration.
    batch_subsample : float or None, default=None
        Fraction of batches to subsample in each iteration.
    threads : int, default=1
        Number of threads for parallel execution.
    kappa : float, default=0.5
        Kappa parameter for optimization.
    tau : float, default=1.0
        Tau parameter for optimization.
    callback : callable or None, default=None
        Callback function to be called during optimization.
    eval_every : int, default=10
        Evaluate model every N iterations.
    verbose : int, default=0
        Verbosity level (0: quiet, >0: increasingly verbose).
    time_limit : float or None, default=None
        Time limit for optimization in seconds.
    test_chroms : tuple, default=("chr1",)
        Chromosomes to use for testing.

    Examples
    --------
    >>> import mutopia as mu
    >>> data = mu.gt.load_dataset("example_data.nc")
    >>> # Create and fit a model with subsampling and 15 components
    >>> model = data.modality().TopographyModel(locus_subsample=0.125, num_components=15)
    >>> model.fit(train_data)
    """

    GT: GTI | MIX

    def __init__(
        self,
        num_components: int = 15,
        init_components: list[str] = [],
        fix_components: list[str] = [],
        seed: int = 0,
        # context model
        context_reg: float = 0.0001,
        context_conditioning: float = 1e-9,
        conditioning_alpha: float = 1e-9,
        # locals model
        pi_prior: float = 1.0,
        tau_prior: float = 1.0,
        estep_iterations: int = 1000,
        difference_tol: float = 5e-5,
        shared_exposures: bool = False,
        # locus model
        locus_model_type: str = "gbt",
        tree_learning_rate: float = 0.15,
        max_depth: int = 5,
        max_trees_per_iter: int = 25,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 30,
        max_features: float = 1.0,
        n_iter_no_change: int = 1,
        use_groups: bool = True,
        add_corpus_intercepts: bool = False,
        convolution_width: int = 0,
        l2_regularization: float = 1,
        max_iter: int = 25,
        init_variance_theta: float = 0.03,
        init_variance_context: float = 0.1,
        # optimization settings
        empirical_bayes: bool = True,
        begin_prior_updates: int = 50,
        stop_condition: int = 50,
        # optimization settings
        num_epochs: int = 2000,
        locus_subsample: Optional[float] = None,
        batch_subsample: Optional[float] = None,
        threads: int = 1,
        kappa: float = 0.5,
        tau: float = 1.0,
        callback: Optional[Callable[..., Any]] = None,
        eval_every: int = 10,
        verbose: int = 0,
        time_limit: Optional[float] = None,
        full_normalizers: bool = True,
        test_chroms: Sequence[str] = ("chr1",),
    ) -> None:

        self.num_components = num_components
        self.init_components = init_components
        self.fix_components = fix_components
        self.seed = seed
        # context model
        self.context_reg = context_reg
        self.context_conditioning = context_conditioning
        self.conditioning_alpha = conditioning_alpha
        # locals model
        self.pi_prior = pi_prior
        self.tau_prior = tau_prior
        self.estep_iterations = estep_iterations
        self.difference_tol = difference_tol
        # locus model
        self.locus_model_type = locus_model_type
        self.tree_learning_rate = tree_learning_rate
        self.max_depth = max_depth
        self.max_trees_per_iter = max_trees_per_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_iter_no_change = n_iter_no_change
        self.use_groups = use_groups
        self.add_corpus_intercepts = add_corpus_intercepts
        self.convolution_width = convolution_width
        self.l2_regularization = l2_regularization
        self.max_iter = max_iter
        self.init_variance_theta = init_variance_theta
        self.init_variance_context = init_variance_context
        # optimization settings
        self.test_chroms = test_chroms
        self.empirical_bayes = empirical_bayes
        self.begin_prior_updates = begin_prior_updates
        self.stop_condition = stop_condition
        self.num_epochs = num_epochs
        self.locus_subsample = locus_subsample
        self.batch_subsample = batch_subsample
        self.threads = threads
        self.kappa = kappa
        self.tau = tau
        self.callback = callback
        self.eval_every = eval_every
        self.verbose = verbose
        self.time_limit = time_limit
        self.shared_exposures = shared_exposures
        self.full_normalizers = full_normalizers

    @property
    def modality(self) -> Any:
        if not hasattr(self, "_modality"):
            raise AttributeError(
                "Modality not set. Please set the modality before using it."
            )
        return get_mode_config(self._modality)

    def sample_params(self, trial: Any, extensive: int = 0) -> Dict[str, Any]:

        params = {}

        if extensive > 0:
            params = {
                "l2_regularization": trial.suggest_float(
                    "l2_regularization", 1e-5, 1000.0, log=True
                ),
                "tree_learning_rate": trial.suggest_float(
                    "tree_learning_rate", 0.025, 0.2
                ),
                "init_variance_theta": trial.suggest_float(
                    "init_variance_theta", 0.025, 0.1
                ),
                "empirical_bayes": trial.suggest_categorical(
                    "empirical_bayes", [True, False]
                ),
            }

        if extensive > 1:
            params["convolution_width"] = trial.suggest_categorical(
                "convolution_width", [0, 1, 2]
            )
            params["max_features"] = 1 / (params["convolution_width"] + 1)

        if extensive > 2:
            params.update(
                {
                    "context_reg": trial.suggest_float(
                        "context_reg", 1e-5, 5e-2, log=True
                    ),
                    "context_conditioning": trial.suggest_float(
                        "context_conditioning", 1e-9, 1e-2, log=True
                    ),
                    "init_variance_context": trial.suggest_float(
                        "init_variance_context", 0.025, 0.15
                    ),
                }
            )

        if extensive > 3:
            params.update(
                {
                    "batch_subsample": trial.suggest_categorical(
                        "batch_subsample",
                        [
                            None,
                            0.0625,
                            0.125,
                            0.25,
                        ],
                    ),
                    "locus_subsample": trial.suggest_categorical(
                        "locus_subsample",
                        [
                            None,
                            0.0625,
                            0.125,
                            0.25,
                        ],
                    ),
                }
            )

        if extensive > 4:
            params.update(
                {
                    "conditioning_alpha": trial.suggest_float(
                        "conditioning_alpha", 1e-10, 1e-7, log=True
                    ),
                }
            )

        return params

    @abstractmethod
    def _init_factor_model(
        self,
        train_corpuses: Sequence[GTensorDataset],
        random_state: "RandomState",
        GT: Any,
        **kw: Any,
    ) -> FactorModel:
        raise NotImplementedError()

    def _init_locals_model(
        self,
        train_datasets: Sequence[GTensorDataset],
        random_state: "RandomState",
    ) -> Any:

        is_sparse = train_datasets[0].X.is_sparse()
        if not all(corpus.X.is_sparse() == is_sparse for corpus in train_datasets):
            raise ValueError(
                "All corpuses must be either sparse or dense - mixing is not allowed!"
            )

        is_mixture = MIX.is_mixture_corpus(train_datasets[0])
        if not all(
            MIX.is_mixture_corpus(corpus) == is_mixture for corpus in train_datasets
        ):
            raise ValueError(
                "All corpuses must be either multi-source or single-source - mixing is not allowed!"
            )

        if is_mixture:
            logger.warning("** Inferring mixture of epigenomes model **")

        if is_mixture:
            locals_model = (
                SparseSharedMixtureModel if is_sparse else DenseSharedMixtureModel
            )
        else:
            locals_model = LDAUpdateSparse if is_sparse else LDAUpdateDense

        # instantiate the locals model
        locals_model = locals_model(
            train_datasets,
            n_components=self.num_components,
            random_state=random_state,
            prior_alpha=self.pi_prior,
            prior_tau=self.tau_prior,
            estep_iterations=self.estep_iterations,
            difference_tol=self.difference_tol,
        )

        return locals_model

    def _train_test_split(
        self, datasets: Sequence[GTensorDataset]
    ) -> Tuple[Tuple[GTensorDataset, ...], Tuple[GTensorDataset, ...]]:
        return list(
            zip(
                *map(
                    lambda dataset: train_test_split(dataset, *self.test_chroms),
                    datasets,
                )
            )
        )

    def init_model(self, train_datasets: Sequence[GTensorDataset]) -> TopographyModel:

        random_state = np.random.RandomState(self.seed)

        self.locals_model_ = self._init_locals_model(
            train_datasets,
            random_state,
        )
        # borrow the GT from the locals model
        self.GT = self.locals_model_.GT

        self.factor_model_ = self._init_factor_model(
            train_datasets,
            random_state,
            self.GT,
            **self.get_params(),
        )

        return self


    def fit(
        self,
        train_datasets: Union[GTensorDataset, Sequence[GTensorDataset]],
        test_datasets: Optional[Union[GTensorDataset, Sequence[GTensorDataset]]] = None,
    ) -> TopographyModel:
        """
        Fit the model to the provided training datasets.

        This method fits the model using a combination of local and factor models.
        If test datasets are not provided, it automatically splits the training
        data into train and test partitions.

        Parameters
        ----------
        train_datasets : GTensorDataset or sequence of GTensorDataset
            One or more datasets to use for training the model.
        test_datasets : GTensorDataset or sequence of GTensorDataset, optional
            Datasets to use for testing the model. If None, a portion of the
            training datasets will be used for testing.

        Returns
        -------
        TopographyModel
            The fitted estimator.

        Notes
        -----
        This method sets the following attributes:
        - ``modality_`` : The modality of the training datasets
        - ``factor_model_`` : The fitted factor model
        - ``locals_model_`` : The fitted locals model
        - ``test_scores_`` : Performance metrics on test datasets
        """
        if not isinstance(train_datasets, (tuple, list)):
            train_datasets = (train_datasets,)
        self._modality = train_datasets[0].attrs["dtype"]

        if not isinstance(test_datasets, (tuple, list)) and not test_datasets is None:
            test_datasets = (test_datasets,)
        elif test_datasets is None or (
            isinstance(test_datasets, (tuple, list)) and len(test_datasets) == 0
        ):
            logger.info("Splitting train/test partitions...")
            train_datasets, test_datasets = self._train_test_split(train_datasets)

        try:
            self.factor_model_, self.locals_model_
        except AttributeError:
            # initialize the models if not already done
            self.init_model(train_datasets)

        n_preset_components = len(self.init_components) + len(self.fix_components)
        self._component_names = (
            self.fix_components
            + self.init_components
            + ["M{}".format(i) for i in range(n_preset_components, self.num_components)]
        )

        random_state = np.random.RandomState(self.seed)

        params = self.get_params()
        params["empirical_bayes"] = self.empirical_bayes and (self.num_components > 1)

        (self.factor_model_, self.locals_model_, self.test_scores_) = _fit_model_impl(
            self.GT,
            train_datasets,
            test_datasets,
            random_state,
            self.factor_model_,
            self.locals_model_,
            **params,
        )

        return self

    @property
    def alpha_(self) -> Any:
        return self.locals_model_.alpha

    @property
    def n_components(self) -> int:
        return self.num_components

    @property
    def component_names(self) -> List[str]:
        try:
            return self._component_names
        except AttributeError:
            return ["M{}".format(i) for i in range(0, self.n_components)]

    def _check_corpus(self, dataset: GTensorDataset, enforce_sample: bool = True) -> None:
        check_corpus(dataset)
        # if enforce_sample:
        #    check_dims(dataset, self.factor_model_)
        dataset["Regions/exposures"] = dataset["Regions/exposures"].astype(
            np.float32, copy=False
        )
        dataset["Regions/context_frequencies"] = dataset[
            "Regions/context_frequencies"
        ].astype(np.float32, copy=False)

    def setup_corpus(self, dataset: GTensorDataset) -> GTensorDataset:
        """
        Set up the corpus dataset with initial state and update normalization factors.

        This method initializes the dataset state using the factor and locals models,
        updates the state from scratch, and then applies normalizers to all expanded datasets.

        Parameters
        ----------
        dataset : GTensorDataset
            The dataset to be set up for modeling

        Returns
        -------
        GTensorDataset
            The initialized and normalized dataset
        """

        logger.info("Setting up dataset state ...")

        dataset = self.GT.init_state(dataset, self.factor_model_, self.locals_model_)

        with ParContext(1) as par:
            self.GT.update_state(
                dataset,
                self.factor_model_,
                from_scratch=True,
                par_context=par,
            )

        logger.info("Done ...")
        return dataset

    def save(self, path: str) -> None:
        """
        Save the model to a file.

        Parameters
        ----------
        path : str
            The file path where the model should be saved.

        Examples
        --------
        >>> model.save("my_model.pkl")
        """

        for model in self.factor_model_.models.values():
            model.prepare_to_save()

        dump(self, path)

    def annot_components(self, dataset: GTensorDataset, normalization: str = "global") -> GTensorDataset:

        spectra = xr.concat(
            [
                self.factor_model_.format_component(component, normalization=normalization)
                for component in range(self.n_components)
            ],
            dim="component",
        )

        if not self.factor_model_.has_interactions:
            component_xrs = {"Spectra/spectra": spectra}
        else:
            interactions, shared_effects = list(
                zip(
                    *[
                        self.factor_model_.format_interactions(component)
                        for component in range(self.n_components)
                    ]
                )
            )

            interactions = xr.concat(
                interactions,
                dim="component",
            )

            shared_effects = xr.concat(
                shared_effects,
                dim="component",
            )

            component_xrs = {
                "Spectra/spectra": spectra,
                "Spectra/interactions": interactions,
                "Spectra/shared_effects": shared_effects,
            }
        

        dataset = dataset.mutate(
            lambda ds: (
                (
                    ds.assign_coords(component=("component", self.component_names))
                    if ds.coords.get("component") is None
                    else ds
                ).assign(
                    **component_xrs
                )
            )
        )
        logger.info(
            "Added keys to dataset: " + ", ".join(component_xrs.keys())
        )
        return dataset

    def annot_contributions(
        self,
        dataset: GTensorDataset,
        threads: int = 1,
        key: str = "contributions",
    ) -> GTensorDataset:
        """
        Calculate and add component contributions to a dataset.

        This method computes the contributions of each component to the dataset using the locals model
        and adds them as a new variable to the dataset.

        Parameters
        ----------
        dataset : GTensorDataset
            Dataset to analyze and annotate with contributions
        threads : int, default=1
            Number of parallel threads to use for computation
        key : str, default="contributions"
            Name of the variable to store contributions in the dataset

        Returns
        -------
        GTensorDataset
            The input dataset with the calculated contributions added as a new variable
            with the specified key name and dimensions ('sample', 'component')

        Notes
        -----
        If the dataset does not have corpus state initialized, this method will
        automatically set it up using setup_corpus().
        """

        self._check_corpus(dataset)

        if not self.GT.has_corpusstate(dataset):
            dataset = self.setup_corpus(dataset)

        contributions = self.locals_model_.predict(
            dataset,
            self.factor_model_,
            threads=threads,
        )

        dataset = dataset.mutate(
            lambda ds: (
                ds.assign_coords(
                    {
                        "component": self.component_names,
                        "source": self.GT.list_sources(dataset),
                    }
                ).assign(
                    {
                        key: contributions,
                    }
                )
            )
        )

        logger.info(f'Added key to dataset: "{key}"')
        return dataset

    def annot_component_distributions(
        self,
        dataset: GTensorDataset,
        threads: int = 1,
        key: str = "component_distributions",
    ) -> GTensorDataset:
        """
        Calculate and add component distributions to a dataset.

        This method computes the probability distributions for each component across
        genomic contexts and adds them as new variables to the dataset.

        Parameters
        ----------
        dataset : GTensorDataset
            Dataset to analyze and annotate with component distributions
        threads : int, default=1
            Number of parallel threads to use for computation
        key : str, default="component_distributions"
            Name of the variable to store distributions in the dataset.
            Per-locus distributions will be stored with the name "{key}_locus"

        Returns
        -------
        GTensorDataset
            The input dataset with the calculated component distributions added as new variables:
            - key: Full distributions with dimensions ('source', 'component', ...)
            - {key}_locus: Per-locus distributions normalized by region length

        Notes
        -----
        If the dataset does not have corpus state initialized, this method will
        automatically set it up using setup_corpus().
        """
        self._check_corpus(dataset)

        if not self.GT.has_corpusstate(dataset):
            dataset = self.setup_corpus(dataset)

        with ParContext(threads) as par:
            X = (
                xr.concat(
                    [
                        self.factor_model_._get_log_mutation_rate_tensor(
                            ds,
                            par_context=par,
                            with_context=False,
                        )
                        for _, ds in self.GT.expand_datasets(dataset)
                    ],
                    dim="source",
                )
                .transpose("source", "component", ...)
                .assign_coords(source=self.GT.list_sources(dataset))
                .pipe(
                    lambda X: np.exp(X - X.max(skipna=True))
                    .fillna(0.0)
                    .astype(np.float32)
                )
            )

        dataset[key] = X

        dataset[f"{key}_locus"] = (
            (X * self.GT.get_freqs(dataset)).sum(
                dim=dims_except_for(X.dims, "source", "locus", "component")
            )
            / self.GT.get_regions(dataset).length
        ).astype(np.float32)

        logger.info(f'Added key: "{key}"')
        logger.info(f'Added key: "{key}_locus"')
        return dataset

    def annot_marginal_prediction(
        self,
        dataset: GTensorDataset,
        threads: int = 1,
        key: str = "predicted_marginal",
    ) -> GTensorDataset:
        """
        Calculate and add marginal predictions to a dataset.

        This method computes marginal mutation rate predictions by marginalizing over
        component distributions weighted by their contributions.

        Parameters
        ----------
        dataset : GTensorDataset
            Dataset to analyze and annotate with marginal predictions
        threads : int, default=1
            Number of parallel threads to use for computation
        key : str, default="predicted_marginal"
            Name of the variable to store marginal predictions in the dataset.
            Per-locus marginal predictions will be stored with the name "{key}_locus"

        Returns
        -------
        GTensorDataset
            The input dataset with marginal predictions added as new variables:
            - key: Marginal mutation rate predictions
            - {key}_locus: Per-locus marginal predictions normalized by region length

        Notes
        -----
        This method requires 'component_distributions' and 'contributions' to be present
        in the dataset. If they are missing, they will be calculated automatically.
        """
        self._check_corpus(dataset, enforce_sample=False)

        if not self.GT.has_corpusstate(dataset):
            dataset = self.setup_corpus(dataset)

        try:
            dataset["component_distributions"]
        except KeyError:
            dataset = self.annot_component_distributions(dataset, threads)

        try:
            dataset["contributions"]
        except KeyError:
            dataset = self.annot_contributions(dataset, threads)

        prior = xr.DataArray(
            self.locals_model_.get_alpha(dataset),
            dims=("component",),
        )
        
        marginal_exposures = (
            dataset["contributions"] + prior
        ).sum(dim="sample")

        marginal = self.factor_model_._log_marginalize_mutrate(
            np.log(dataset["component_distributions"]),
            marginal_exposures
        )

        dataset[key] = (
            np.exp(marginal - marginal.max(skipna=True))
            .fillna(0.0).astype(np.float32)
        )
        dataset[f"{key}_locus"] = (
            (np.exp(marginal) * dataset["Regions/context_frequencies"]).sum(
                dim=dims_except_for(marginal.dims, "source", "locus")
            )
            / dataset["Regions/length"]
        ).astype(np.float32)

        logger.info(f'Added key: "{key}"')
        logger.info(f'Added key: "{key}_locus"')

        return dataset

    def annot_SHAP_values(
        self,
        dataset: GTensorDataset,
        *components: Union[int, str],
        threads: int = 1,
        scan: bool = False,
        n_samples: int = 2000,
        seed: int = 42,
        key: str = "SHAP_values",
        source: Optional[str] = None,
    ) -> GTensorDataset:
        """
        Calculate and add SHAP values to explain component predictions.

        This method uses SHAP (SHapley Additive exPlanations) to compute feature
        importance values for understanding how genomic features contribute to
        component predictions.

        Parameters
        ----------
        dataset : GTensorDataset
            Dataset to analyze and annotate with SHAP values
        *components : int or str
            Component indices or names to calculate SHAP values for. If none provided,
            calculates for all components.
        threads : int, default=1
            Number of parallel threads to use for computation
        scan : bool, default=False
            If True, calculates SHAP values for all loci. If False, subsamples loci.
        n_samples : int, default=2000
            Number of loci to subsample when scan=False
        seed : int, default=42
            Random seed for reproducible subsampling
        key : str, default="SHAP_values"
            Name of the variable to store SHAP values in the dataset

        Returns
        -------
        GTensorDataset
            The input dataset with SHAP values added as a new variable with the specified
            key name and dimensions ('shap_component', 'locus' or 'shap_locus', 'shap_features')

        Raises
        ------
        ImportError
            If the SHAP library is not installed

        Notes
        -----
        This method requires the SHAP library to be installed. Install with:
        pip install shap
        """

        if self.GT.is_mixture_corpus(dataset) and source is None:
            raise ValueError("If you're working with a mixture corpus, you must choose a 'source' celltype from which to annotate SHAP values")

        try:
            import shap
        except ImportError:
            raise ImportError("SHAP is required to calculate SHAP values")

        self._check_corpus(dataset, enforce_sample=False)

        if not self.GT.has_corpusstate(dataset):
            dataset = self.setup_corpus(dataset)

        if not scan:
            subset_loci = np.random.RandomState(seed).choice(
                dataset.locus.size, max(n_samples, 50), replace=False
            )
            subset = dataset.isel(locus=subset_loci)
        else:
            subset = dataset

        n_loci = subset.sizes["locus"]
        locus_model = self.factor_model_.models["theta_model"]

        if self.GT.is_mixture_corpus(dataset):
            subset = self.GT.fetch_source(subset, source)

        X = locus_model._fetch_feature_matrix(subset)

        background_idx = np.random.RandomState(0).choice(
            len(X), size=min(1000, len(X)), replace=False
        )

        def _component_shap(k):

            logger.info(f"Calculating SHAP values for {self.component_names[k]} ...")

            shaps = shap.TreeExplainer(
                locus_model.rate_models[k],
                X[background_idx],
            ).shap_values(
                X,
                check_additivity=False,
                approximate=False,
            )

            return np.squeeze(shaps)

        use_components = list(
            components if not len(components) == 0 else list(range(self.n_components))
        )

        shap_matrix = np.array(
            parallel_map(
                (partial(_component_shap, k) for k in use_components), threads=threads
            )
        )

        features = dataset.coords["feature"].data
        coords = {
            "shap_features": features,
            "shap_component": [self.component_names[k] for k in use_components],
        }

        if not scan:
            coords["shap_locus"] = subset.locus.data

        shap_matrix = shap_matrix.reshape((len(use_components), n_loci, -1))

        dataset[key] = xr.DataArray(
            shap_matrix,
            dims=(
                "shap_component",
                "locus" if scan else "shap_locus",
                "shap_features",
            ),
            coords=coords,
        )

        logger.info(f'Added key: "{key}"')
        return dataset

    def annot_data(
        self, 
        dataset: GTensorDataset, 
        subset_region: Optional[str] = None,
        threads: int = 1, 
        source: Optional[str] = None, 
        calc_shap=True
    ) -> GTensorDataset:
        """
        Annotate a dataset with comprehensive model analysis information.

        This method applies a series of annotation functions to enrich the dataset with
        various types of model-derived insights including component analysis, contribution
        calculations, SHAP values, component distributions, and marginal predictions.

        Args:
            dataset: The input dataset to be annotated with model analysis information.
            threads (int, optional): Number of threads to use for parallel processing
                in annotation functions that support multithreading. Defaults to 1.

        Returns:
            The annotated dataset containing the original data plus all computed
            annotations from the applied annotation functions.

        Note:
            The annotation functions are applied sequentially in the following order:
            1. Component annotations
            2. Contribution annotations
            3. SHAP value annotations
            4. Component distribution annotations
            5. Marginal prediction annotations
        """
        from mutopia.gtensor import annot_empirical_marginal, slice_regions
        
        _annot_fns = [
            self.setup_corpus,
            partial(self.annot_contributions, threads=threads),
            lambda ds: slice_regions(ds, subset_region) if subset_region else ds,
            self.annot_components,
            partial(self.annot_SHAP_values, threads=threads, source=source) if calc_shap else lambda ds: ds,
            partial(self.annot_component_distributions, threads=threads),
            partial(self.annot_marginal_prediction, threads=threads),
            annot_empirical_marginal,
        ]

        for fn in _annot_fns:
            dataset = fn(dataset)

        return dataset
