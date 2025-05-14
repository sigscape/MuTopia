import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import trange
from functools import partial
from joblib import dump, delayed
from collections import defaultdict
from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod
import typing
from ..utils import *
from .model_components import *
from .latent_var_models import *
from ..plot.coef_matrix_plot import _plot_interaction_matrix
from ..gtensor import *
from ..tuning import sample_params
from .optim import fit_model
from .factor_model import FactorModel
from .latent_var_models.base import LocalsModel
from ..mixture_model import SparseMixtureModel, DenseMixtureModel
# interfaces
from . import gtensor_interface as CS
from ..mixture_model import mixture_interface as MIX

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

    def sample_params(self, study, trial, extensive=0):
        return sample_params(study, trial, extensive=extensive)
    
    @abstractmethod
    def _init_factor_model(
        self,
        train_corpuses,
        random_state,
        GT,
        **kw,
    ) -> FactorModel:
        raise NotImplementedError()

    def _choose_locals_model(
        self,
        is_mixture=True,
        is_sparse=True,
    ):
        if is_mixture and is_sparse:
            return SparseMixtureModel
        elif is_mixture and not is_sparse:
            raise NotImplementedError()
        elif not is_mixture and is_sparse:
            return LDAUpdateSparse
        else:
            return LDAUpdateDense


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
            raise ValueError("All corpuses must be either sparse or dense - mixing is not allowed!")
        
        is_mixture = MIX.is_mixture_corpus(train_datasets[0])
        if not all(
            MIX.is_mixture_corpus(corpus) == is_mixture
            for corpus in train_datasets + test_datasets
        ):
            raise ValueError("All corpuses must be either multi-source or single-source - mixing is not allowed!")
        
        if is_mixture:
            logger.warning("** Inferring mixture of epigenomes model **")

        locals_model = self._choose_locals_model(
            is_mixture=is_mixture, 
            is_sparse=is_sparse
        )(
            (MIX if is_mixture else CS),
            train_datasets,
            n_components=self.num_components,
            random_state=random_state,
            prior_alpha=self.pi_prior,
            prior_tau=self.tau_prior,
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

    def rename_components(self, dataset, names: typing.List[str]):
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

        for model in self.factor_model_.models.values():
            model.prepare_to_save()

        dump(self, path)

    def plot_signature(self, component, *select, normalization="global", **kwargs):
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
        ncols=3,
        normalization="global",
        width=3.5,
        height=1.25,
        show=True,
        **kwargs,
    ):

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
