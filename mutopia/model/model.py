import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import trange
from functools import partial
from joblib import dump, delayed
from collections import defaultdict
from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod
from typing import *
from ..utils import *
from .model_components import *
from .latent_var_models import *
from ..plot.coef_matrix_plot import _plot_interaction_matrix
from ..gtensor import *
from .optim import fit_model
from .factor_model import FactorModel
from ..mixture_model import *

# interfaces
from .gtensor_interface import GtensorInterface as CS
from ..mixture_model.mixture_interface import MixtureInterface as MIX

"""
The Model class is a wrapper around a trained model state object, 
and provides the high-level interface for interacting with the model.
This is the entry point for the user to interact with and annotate data.
"""


class TopographyModel(ABC, BaseEstimator):

    def __init__(
        self,
        num_components=15,
        init_components=[],
        fix_components=[],
        seed=0,
        # context model
        context_reg=0.0001,
        context_conditioning=1e-9,
        conditioning_alpha=1e-9,
        # locals model
        pi_prior=1.0,
        tau_prior=1.0,
        estep_iterations=1000,
        difference_tol=5e-5,
        shared_exposures=False,
        # locus model
        locus_model_type="gbt",
        tree_learning_rate=0.15,
        max_depth=5,
        max_trees_per_iter=25,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        max_features=1.0,
        n_iter_no_change=1,
        use_groups=True,
        add_corpus_intercepts=False,
        convolution_width=0,
        l2_regularization=1,
        max_iter=25,
        init_variance_theta=0.03,
        init_variance_context=0.1,
        # optimization settings
        empirical_bayes=True,
        begin_prior_updates=50,
        stop_condition=50,
        # optimization settings
        num_epochs=2000,
        locus_subsample=None,
        batch_subsample=None,
        threads=1,
        kappa=0.5,
        tau=1.0,
        callback=None,
        eval_every=10,
        verbose=0,
        time_limit=None,
        test_chroms=("chr1",),
    ):
        """Initialize the signature model.

        This constructor sets up a signature model with specified parameters for components, regularization,
        conditioning, and optimization settings.

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
        >>> model = TopographyModel(locus_subsample=0.125, num_components=15)
        >>> model.fit(train_data)
        >>>
        >>> # Annotate contributions of each component to the data
        >>> annotated_data = model.annot_contributions(train_data)
        >>>
        >>> # Visualize the first signature
        >>> model.plot_signature(0)
        """

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

    def sample_params(self, study, trial, extensive=0):

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
        train_corpuses,
        random_state,
        GT,
        **kw,
    ) -> FactorModel:
        raise NotImplementedError()

    def _init_locals_model(
        self,
        train_datasets,
        test_datasets,
        random_state,
    ):

        is_sparse = train_datasets[0].X.is_sparse()
        if not all(
            corpus.X.is_sparse() == is_sparse
            for corpus in train_datasets + test_datasets
        ):
            raise ValueError(
                "All corpuses must be either sparse or dense - mixing is not allowed!"
            )

        is_mixture = MIX.is_mixture_corpus(train_datasets[0])
        if not all(
            MIX.is_mixture_corpus(corpus) == is_mixture
            for corpus in train_datasets + test_datasets
        ):
            raise ValueError(
                "All corpuses must be either multi-source or single-source - mixing is not allowed!"
            )

        if is_mixture:
            logger.warning("** Inferring mixture of epigenomes model **")

        if is_mixture:
            locals_model = type(
                "MixtureLocalsModel",
                (
                    SparseMixtureModel if is_sparse else DenseMixtureModel,
                    (
                        SharedExposuresMixtureModel
                        if self.shared_exposures
                        else DifferentExposuresMixtureModel
                    ),
                ),
                {},
            )
        else:
            locals_model = LDAUpdateSparse if is_sparse else LDAUpdateDense

        # instantiate the locals model
        locals_model = locals_model(
            (MIX if is_mixture else CS),
            train_datasets,
            n_components=self.num_components,
            random_state=random_state,
            prior_alpha=self.pi_prior,
            prior_tau=self.tau_prior,
            estep_iterations=self.estep_iterations,
            difference_tol=self.difference_tol,
        )

        return locals_model

    def _train_test_split(self, datasets):
        return list(
            zip(
                *map(
                    lambda dataset: train_test_split(dataset, *self.test_chroms),
                    datasets,
                )
            )
        )

    def fit(
        self,
        *train_datasets,
        test_datasets=None,
    ):
        """
        Fit the model to the provided training datasets.

        This method fits the model using a combination of local and factor models.
        If test datasets are not provided, it automatically splits the training
        data into train and test partitions.

        Parameters
        ----------
        *train_datasets : list of Dataset
            One or more datasets to use for training the model.
        test_datasets : list of Dataset, optional
            Datasets to use for testing the model. If None, a portion of the
            training datasets will be used for testing.

        Returns
        -------
        self : object
            The fitted estimator.

        Raises
        ------
        ValueError
            If no training datasets are provided.

        Notes
        -----
        This method sets the following attributes:
        - modality_ : The modality of the training datasets
        - factor_model_ : The fitted factor model
        - locals_model_ : The fitted locals model
        - test_scores_ : Performance metrics on test datasets
        """

        self.modality_ = train_datasets[0].modality()

        if len(train_datasets) == 0:
            raise ValueError("At least one dataset is required to fit the model.")

        if test_datasets is None:
            logger.info("Splitting train/test partitions...")
            train_datasets, test_datasets = self._train_test_split(train_datasets)

        random_state = np.random.RandomState(self.seed)

        locals_model = self._init_locals_model(
            train_datasets,
            test_datasets,
            random_state,
        )
        # borrow the GT from the locals model
        self.GT = locals_model.GT

        factor_model = self._init_factor_model(
            train_datasets,
            random_state,
            self.GT,
            **self.get_params(),
        )

        (self.factor_model_, self.locals_model_, self.test_scores_) = fit_model(
            self.GT,
            train_datasets,
            test_datasets,
            random_state,
            factor_model,
            locals_model,
            **self.get_params(),
        )

        return self

    @property
    def n_components(self):
        return self.num_components

    @property
    def component_names(self):
        try:
            return self._component_names
        except AttributeError:
            return ["M{}".format(i) for i in range(0, self.n_components)]

    def _get_k(self, component_name):
        if isinstance(component_name, int):
            return component_name

        try:
            return self.component_names.index(component_name)
        except ValueError:
            raise ValueError(f"Component {component_name} not found in model.")

    def rename_components(self, dataset, names: List[str]):
        """
        Rename the components of the model and update the dataset coordinates accordingly.

        Parameters
        ----------
        dataset : xarray.Dataset
            The dataset containing model components to be renamed.
        names : typing.List[str]
            New names for the components. Must have the same length as the number of components in the model.

        Returns
        -------
        xarray.Dataset
            The dataset with updated component names in coordinates.

        Raises
        ------
        ValueError
            If the number of provided names doesn't match the number of components.
        KeyError
            If some components in the dataset's "shap_component" coordinate don't match the model components.

        Notes
        -----
        This method also updates the internal _component_names attribute of the model.
        """
        if not len(names) == self.n_components:
            raise ValueError("The number of names must match the number of components")

        name_map = dict(zip(self.component_names, names))
        new_coords = {"component": names}

        if "shap_component" in dataset.coords:
            try:
                new_coords["shap_component"] = [
                    name_map[c] for c in dataset.coords["shap_component"].data
                ]
            except KeyError:
                raise KeyError(
                    "Some components in dataset do not match the model components. Just delete the SHAP_values and try again."
                )

        dataset = dataset.assign_coords(new_coords)
        self._component_names = names

        return dataset

    def _check_corpus(self, dataset, enforce_sample=True):
        check_corpus(dataset, enforce_sample=enforce_sample)
        if enforce_sample:
            check_dims(dataset, self.model_state_)

    def setup_corpus(self, dataset):
        """
        Set up the corpus dataset with initial state and update normalization factors.

        This method initializes the dataset state using the factor and locals models,
        updates the state from scratch, and then applies normalizers to all expanded datasets.

        Parameters
        ----------
        dataset : Dataset
            The dataset to be set up for modeling

        Returns
        -------
        Dataset
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

        for ds in self.GT.expand_datasets(dataset):
            self.GT.update_normalizers(ds, self.factor_model_.get_normalizers(ds))

        logger.info("Done ...")
        return dataset

    def save(self, path):
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

    def plot_signature(self, component, *select, normalization="global", **kwargs):
        """
        Plots the signature for a given component.
        Parameters
        ----------
        component : int or str
            The component index or name to plot.
        *select : str, optional
            Mesoscale selection strings to plot. If none are provided, defaults to ["Baseline"].
        normalization : {global, weighted, none}, default="global"
            The normalization method to use for the signature.
        **kwargs : dict
            Additional keyword arguments to pass to the underlying plot method.
        Returns
        -------
        matplotlib.figure.Figure or other plot object
            The resulting plot object returned by the modality's plot method.
        Notes
        -----
        This method retrieves the component by name or index, formats the signature using the factor model,
        and delegates the actual plotting to the modality's plot method.
        """

        if len(select) == 0:
            select = ["Baseline"]

        component = self._get_k(component)

        return self.modality_.plot(
            self.factor_model_.format_signature(component, normalization=normalization),
            *select,
            **kwargs,
        )

    def signature_report(
        self,
        component,
        normalization="global",
        width=5.25,
        height=2.0,
        show=True,
    ):
        """
        Generate a comprehensive report for a specific signature component.

        This method creates a figure with signature plots for mesoscale states and an interaction matrix
        for the specified component, providing a visual representation of the signature's characteristics.

        Parameters
        ----------
        component : int or str
            The signature component to visualize. Can be an integer index or a string identifier.
        normalization : str, default="global"
            The normalization method to use for the signatures.
        width : float, default=5.25
            The base width of the figure in inches. The actual figure width may be adjusted based on the number of states.
        height : float, default=2.0
            The base height per signature group in inches.
        show : bool, default=True
            Whether to display the figure immediately.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure containing signature plots and interaction matrix.

        Notes
        -----
        The report organizes mesoscale states into groups based on their prefix (before the colon),
        and displays them in separate rows. For singleton state groups (except Baseline),
        the Baseline state is automatically added as a reference.
        """

        component = self._get_k(component)

        signatures = self.factor_model_.format_signature(
            component, normalization=normalization
        )
        n_rows = len(signatures.mesoscale_state)

        state_groups = defaultdict(list)
        for state in signatures.mesoscale_state.values:
            state_groups[state.split(":")[0]].append(state)

        for k, v in state_groups.items():
            if not k == "Baseline" and len(v) == 1:
                state_groups[k].append("Baseline")

        max_n_states = max(map(len, state_groups.values()))
        n_sigs = len(state_groups)
        fig = plt.figure(figsize=(max(width * max_n_states, 10), height * n_sigs + 3))

        gs = fig.add_gridspec(
            2,
            1,
            height_ratios=[height * n_sigs, 1 + 0.35 * n_rows],
            hspace=0.1,
        )

        gs0 = gs[0].subgridspec(
            n_sigs + 1,
            max_n_states,
            hspace=0.75,
            wspace=0.5,
            width_ratios=[3] + [1] * (max_n_states - 1),
        )

        for i, states in enumerate(state_groups.values()):
            ax = fig.add_subplot(gs0[i, : len(states)])
            self.plot_signature(
                component,
                *states,
                ax=ax,
            )

        self.plot_interaction_matrix(
            component,
            gridspec=gs[1],
            normalization=normalization,
        )

        fig.suptitle(f"Signature {component} report", fontsize=12, y=0.95)

        if show:
            plt.show()

        return fig

    def plot_interaction_matrix(
        self,
        component,
        palette=diverging_palette,
        gridspec=None,
        normalization="global",
        **kw,
    ):
        """
        Generate a visualization of component interactions.

        This method creates a plot showing the interaction matrix for a specified component.
        It displays shared effects and context-specific interactions for genomic signatures.

        Parameters
        ----------
        component : int or str
            The component index or identifier to visualize.
        palette : function, optional
            A color palette function to use for visualization, defaults to diverging_palette.
        normalization : str, optional
            Method for normalizing the signature values.
        **kw : dict
            Additional keyword arguments passed to the underlying plotting function.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the visualization of the interaction matrix.

        Notes
        -----
        The interaction matrix shows how the component behaves across different contexts,
        highlighting both shared effects and context-specific variations.
        """

        component = self._get_k(component)

        flatten = partial(self.modality_._flatten_observations)

        interactions = self.factor_model_.format_interactions(component)

        shared_effects = interactions.sel(context="Shared effect").to_pandas()
        interactions = flatten(
            interactions.drop_sel(context="Shared effect")
        ).to_pandas()

        signature = self.factor_model_.format_signature(
            component, normalization=normalization
        )

        return _plot_interaction_matrix(
            partial(self.modality_.plot, signature, "Baseline"),
            interactions,
            shared_effects,  # .iloc[:,0],
            palette=palette,
            gridspec=gridspec,
            **kw,
        )

    def signature_panel(
        self,
        ncols=4,
        normalization="global",
        width=3.5,
        height=1.25,
        show=True,
        **kwargs,
    ):
        """
        Create a panel of signature plots for all components in the model.

        Parameters
        ----------
        ncols : int, default=3
            Number of columns in the panel.
        normalization : str, default="global"
            Normalization method for the signatures.
        width : float, default=3.5
            Width of each subplot in inches.
        height : float, default=1.25
            Height of each subplot in inches.
        show : bool, default=True
            If True, displays the figure. If False, returns the figure object.
        **kwargs
            Additional keyword arguments passed to plot_signature method.

        Returns
        -------
        fig : matplotlib.figure.Figure, optional
            The figure object containing the panel of signatures. Only returned if show=False.

        Notes
        -----
        This method creates a grid of subplots, each displaying one signature component.
        The number of rows is calculated based on ncols and the number of components.
        Component names are displayed as y-axis labels.
        """

        K = self.n_components
        nrows = int(np.ceil(K / ncols))

        fig, ax = plt.subplots(
            nrows,
            ncols,
            figsize=(width * ncols, height * nrows),
            gridspec_kw={"hspace": 0.5, "wspace": 0.25},
        )

        for k in range(self.n_components):
            _ax = np.ravel(ax)[k]
            self.plot_signature(k, ax=_ax, normalization=normalization, **kwargs)
            _ax.set_ylabel(self.component_names[k], fontsize=8)

        for _ax in np.ravel(ax)[self.n_components :]:
            _ax.axis("off")

        if show:
            plt.show()
        else:
            return fig

    def annot_contributions(
        self,
        dataset,
        threads=1,
        verbose=0,
    ):
        """
        Calculate and add component contributions to a dataset.

        This method computes the contributions of each component to the dataset using the locals model
        and adds them as a new variable to the dataset.

        Parameters
        ----------
        dataset : xarray.Dataset or str
            Dataset or name of dataset to analyze
        threads : int, default=1
            Number of parallel threads to use for computation
        verbose : int, default=0
            Verbosity level for parallel computation

        Returns
        -------
        xarray.Dataset
            The input dataset with the calculated contributions added as a new variable
            with dimensions ('sample', 'component')
        """

        self._check_corpus(dataset)

        if not self.GT.has_corpusstate(dataset):
            dataset = self.setup_corpus(dataset)

        with ParContext(threads, verbose=verbose) as par:
            contributions = self.locals_model.predict(
                dataset, self.factor_model_, par_context=par
            )

        dataset = CorpusInterface(dataset)
        dataset.dataset = dataset.assign_coords(
            {
                "component": self.component_names,
            }
        ).assign(
            {
                "contributions": contributions,
            }
        )

        logger.info('Added key to dataset: "contributions"')
        return dataset

    def annot_component_distributions(
        self,
        dataset,
        threads=1,
    ):
        self._check_corpus(dataset, enforce_sample=False)

        if not self.GT.has_corpusstate(dataset):
            dataset = self.setup_corpus(dataset)

        with ParContext(threads) as par:
            lmrt = self.factor_model_._get_log_mutation_rate_tensor(
                dataset,
                par_context=par,
                with_context=False,
            )

        dataset["component_distributions"] = (
            np.exp(lmrt - lmrt.max(skipna=True)).fillna(0.0).astype(np.float32)
        )

        dataset["component_locus_distributions"] = (
            (dataset.component_distributions * dataset.regions.context_frequencies).sum(
                dim=dims_except_for(
                    dataset.component_distributions.dims, "locus", "component"
                )
            )
            / dataset.regions.length
        ).astype(np.float32)

        logger.info('Added key: "component_distributions"')
        logger.info('Added key: "component_locus_distributions"')
        return dataset

    def annot_marginal_prediction(
        self,
        dataset,
        exposures=None,
        threads=1,
    ):
        self._check_corpus(dataset, enforce_sample=False)

        if not self.GT.has_corpusstate(dataset):
            dataset = self.setup_corpus(dataset)

        try:
            dataset["component_distributions"]
        except KeyError:
            dataset = self.annot_component_distributions(dataset, threads)

        if exposures is None:
            try:
                dataset["contributions"]
            except KeyError:
                dataset = self.annot_contributions(dataset, threads)
            marginal_exposures = dataset["contributions"].sum(dim="sample").data
        else:
            marginal_exposures = np.sum(
                [
                    exposures(dataset, sample_name)
                    for sample_name in self.GT.list_samples(dataset)
                ],
                axis=0,
            )

        marginal = self.factor_model_._log_marginalize_mutrate(
            np.log(dataset["component_distributions"]), marginal_exposures
        )

        dataset["predicted_marginal"] = (
            np.exp(marginal - marginal.max(skipna=True)).fillna(0.0).astype(np.float32)
        )
        dataset["predicted_locus_marginal"] = (
            (np.exp(marginal) * dataset.regions.context_frequencies).sum(
                dim=dims_except_for(marginal.dims, "locus")
            )
            / dataset.regions.length
        ).astype(np.float32)

        logger.info('Added key: "predicted_marginal"')
        logger.info('Added key: "predicted_locus_marginal"')

        return dataset

    def annot_SHAP_values(
        self,
        dataset,
        *components,
        threads=1,
        scan=False,
        n_samples=2000,
        seed=42,
    ):

        try:
            import shap
        except ImportError:
            raise ImportError("SHAP is required to calculate SHAP values")

        self._check_corpus(dataset, enforce_sample=False)

        if not self.GT.has_corpusstate(dataset):
            dataset = self.setup_corpus(dataset)

        if not scan:
            subset_loci = np.random.RandomState(seed).choice(
                dataset.locus.size, max(n_samples, 1500), replace=False
            )
            subset = dataset.isel(locus=subset_loci)
        else:
            subset = dataset

        locus_model = self.factor_model_.models["theta_model"]
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
            map(
                self._get_k,
                (
                    components
                    if not len(components) == 0
                    else list(range(self.n_components))
                ),
            )
        )

        with ParContext(threads) as par:
            shap_matrix = np.array(
                list(par(delayed(_component_shap)(k) for k in use_components))
            )

        features = dataset.state.coords["feature"].data
        coords = {
            "shap_features": features,
            "shap_component": [self.component_names[k] for k in use_components],
        }

        if not scan:
            coords["shap_locus"] = subset.locus.data

        dataset["SHAP_values"] = xr.DataArray(
            shap_matrix,
            dims=("shap_component", "locus" if scan else "shap_locus", "shap_features"),
            coords=coords,
        )

        logger.info('Added key: "SHAP_values"')
        return dataset

    def score(
        self,
        dataset,
        exposures=None,
        threads=1,
    ):
        self._check_corpus(dataset)

        if not self.GT.has_corpusstate(dataset):
            dataset = self.setup_corpus(dataset)

        with ParContext(threads) as par:
            return self.locals_model_.deviance(
                self.factor_model_,
                (dataset,),
                exposures_fn=(
                    using_exposures_from(dataset) if exposures is None else exposures
                ),
                par_context=par,
            ).item()

    def marginal_ll(
        self,
        dataset,
        threads=1,
        alpha=None,
        steps=64000,
    ):

        if isinstance(alpha, str):
            if alpha == "uniform":
                alpha = np.ones(self.n_components)
            else:
                raise ValueError('Alpha must be a list of floats or "uniform"')

        self._check_corpus(dataset)

        if not self.GT.has_corpusstate(dataset):
            dataset = self.setup_corpus(dataset)

        ll_fns = self.locals_model.get_marginal_ll_fns(
            (dataset,),
            self.factor_model_,
            alpha=alpha,
            steps=steps,
        )

        bar = trange(
            len(dataset.list_samples()), desc="Calculating marginal lls", leave=False
        )

        ll = 0.0
        for sample_ll, *_ in parallel_gen(ll_fns, threads, ordered=False):
            ll += sample_ll
            bar.update(1)

        return ll

    def format_signature(self, component, normalization="global"):

        component = self._get_k(component)

        return self.modality_._flatten_observations(
            self.factor_model_.format_signature(component, normalization=normalization)
        )

    @property
    def alpha_(self):
        return self.locals_model_.alpha

    def execel_report(self, dataset, output):

        try:
            from pandas import ExcelWriter
        except ImportError:
            raise ImportError(
                "openpyxl is required to save excel reports, install with `pip install openpyxl`"
            )

        renorm = lambda x: x / x.sum() * 1000

        with ExcelWriter(output) as writer:

            for sig in self.component_names:
                (
                    renorm(self.format_signature(sig))
                    .to_pandas()
                    .T.to_excel(
                        writer,
                        sheet_name=f"Signature_{sig}",
                    )
                )

            if hasattr(dataset, "contributions"):
                (
                    dataset.contributions.to_pandas().to_excel(
                        writer,
                        sheet_name="Sample_contributions",
                    )
                )

            if hasattr(dataset, "SHAP_values"):

                shap_components = dataset.SHAP_values.coords["shap_component"].values
                expl = get_explanation(dataset, shap_components[0])

                DataFrame(
                    expl.data,
                    columns=expl.feature_names,
                ).to_excel(
                    writer,
                    sheet_name="SHAP_transformed_features",
                    index=False,
                )

                display_data = expl.display_data.copy()
                display_data.columns = expl.feature_names
                display_data.to_excel(
                    writer,
                    sheet_name="SHAP_original_features",
                    index=False,
                )

                for component in shap_components:

                    expl = get_explanation(dataset, component)

                    DataFrame(
                        expl.values,
                        columns=expl.feature_names,
                    ).to_excel(
                        writer,
                        sheet_name="SHAP_values_{}".format(component),
                        index=False,
                    )
