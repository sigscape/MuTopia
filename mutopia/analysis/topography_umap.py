"""Project MuTopia model components into a reference topography UMAP space.

A MuTopia component's *topography* is its normalized mutation-rate profile across
genomic bins -- ``component_distributions_locus``, written by
:meth:`mutopia.model.base.TopographyModel.annot_component_distributions`.  Two
components trained on different cohorts can be compared directly in that space,
which is what lets signatures be matched across tumor types.

:class:`TopographyUMAP` freezes such a space: a reference matrix of components, a
2D UMAP layout of them, and their curated annotations.  New components are placed
into the same layout with :meth:`~TopographyUMAP.transform`.

    >>> import mutopia.analysis as mu
    >>> ref = mu.load_reference_umap()
    >>> data = mu.gt.load_dataset("my_cohort.annotated.nc", with_samples=False)
    >>> ref.transform(data)
    >>> ref.kneighbors(data, k=5)

Requires the optional ``umap-learn`` dependency (``pip install mutopia[umap]``).

Notes
-----
The 2D coordinates are for display.  UMAP's ``transform`` is an approximate
re-optimization of new points against a frozen reference layout, and a reference
built from a few hundred components is sparse.  The nearest-neighbour table and
the cluster vote -- both computed in the full-dimensional cosine space, not in the
embedding -- are the defensible readouts.  Always check ``is_outlier`` /
``outlier_score``: a component unlike anything in the reference is still assigned
a plausible-looking coordinate and cluster.
"""

import json
import logging
import os
import urllib.request
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

__all__ = ["TopographyUMAP", "load_reference_umap", "load_reference_coordinates",
           "annot_component_rates", "REFERENCE_ARTIFACT", "REFERENCE_COORDINATES"]


#: Rows of the feature matrix are normalized to sum to this value.  Cosine
#: distance is scale-invariant so the constant is cosmetic for the fit, but it
#: keeps the stored matrix inside float16's normal range (values ~1e0 rather than
#: ~1e-5, which would fall into the subnormal band and lose precision).
_ROW_SCALE = 100_000.0

_LOCUS_KEY = "component_distributions_locus"

REFERENCE_ARTIFACT = os.path.join(
    os.path.dirname(__file__), "reference", "pancan_topography_umap.npz"
)

#: Canonical layout only -- component, annotations, UMAP1, UMAP2.  A few tens of
#: kB, no feature matrix, and no umap-learn needed to read it.  Ships in the wheel.
REFERENCE_COORDINATES = os.path.join(
    os.path.dirname(__file__), "reference", "pancan_topography_umap.coords.tsv"
)

#: The feature matrix is ~11 MB, so it is a release asset rather than a committed
#: file -- the same arrangement `tests/conftest.py` uses for the test fixtures.
#: Update both when the artifact is rebuilt (see DEVELOPING.md).
ARTIFACT_RELEASE_TAG = "topography-umap-reference"
ARTIFACT_BASE_URL = (
    f"https://github.com/sigscape/MuTopia/releases/download/{ARTIFACT_RELEASE_TAG}"
)


def _cache_path(name):
    """Where a downloaded artifact is cached.

    Never inside the installed package: site-packages is often read-only, and
    writing there would make the install differ from what was shipped.
    """
    root = os.environ.get("MUTOPIA_CACHE_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "mutopia"
    )
    return os.path.join(root, name)


def _ensure_artifact(path=None, download=True):
    """Resolve the reference artifact, downloading it once if needed."""
    if path is not None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No reference artifact at {path}")
        return path

    name = os.path.basename(REFERENCE_ARTIFACT)
    for candidate in (REFERENCE_ARTIFACT, _cache_path(name)):
        if os.path.exists(candidate):
            return candidate

    cached = _cache_path(name)
    url = f"{ARTIFACT_BASE_URL}/{name}"
    if not download:
        raise FileNotFoundError(
            f"Reference artifact not found locally and download=False. "
            f"Fetch {url} to {cached}, or build it with "
            f"`python tools/build_reference_artifact.py --help`."
        )

    os.makedirs(os.path.dirname(cached), exist_ok=True)
    tmp = cached + ".part"
    logger.info("downloading reference artifact from %s", url)
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, cached)
    except Exception as err:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise FileNotFoundError(
            f"Could not download the reference artifact from {url} ({err}). "
            f"Download it manually to {cached}, or rebuild it with "
            f"`python tools/build_reference_artifact.py --help`."
        ) from err
    logger.info("cached reference artifact at %s (%.1f MB)",
                cached, os.path.getsize(cached) / 1e6)
    return cached

_METADATA_FIELDS = ("tumor_type", "class", "cluster_id", "cluster_name")


def _require_umap():
    try:
        import umap
    except ImportError as err:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "TopographyUMAP requires the optional `umap-learn` dependency. "
            "Install it with `pip install mutopia[umap]`."
        ) from err
    return umap


def annot_component_rates(model, dataset, source=None, threads=1, key=_LOCUS_KEY):
    """Evaluate a trained model's component locus rates on *any* gTensor's axis.

    ``model.annot_component_distributions`` routes through ``setup_corpus``, which
    initializes the *locals* model and therefore iterates the dataset's samples.
    Datasets written with ``write_samples=False`` have no ``raw`` group, so that
    path raises ``ValueError: Sample ... not found`` — for every model, not just
    unusual ones.  Component locus distributions depend only on the factor model
    (context + theta), so this initializes just that half and skips the
    sample-dependent step.

    The point of doing this is comparability.  A model is a function of genomic
    features, so evaluating it on the *reference* gTensor puts its components on
    the reference locus axis — which lets :meth:`TopographyUMAP.transform` take
    the strict path instead of the lossy aggregate one, even when the model was
    trained on a completely different regions bed.

    Parameters
    ----------
    model : TopographyModel
        A trained model, e.g. from ``mu.load_model(...)``.
    dataset : gTensor Dataset
        The axis to evaluate on.  Load it with ``with_state=False`` so the
        previous model's corpus state is shed (``mu.gt.load_dataset(path,
        with_samples=False, with_state=False)``).
    source : str, optional
        Cell type / source for multi-source models.

    Returns
    -------
    The dataset with ``component_distributions_locus`` added, dims
    ``(component, locus)`` and ``component`` coordinates taken from
    ``model.component_names``.  Suitable for
    :meth:`TopographyUMAP.transform` and for ``mutopia.plot.track_plot``.
    """
    from mutopia.gtensor.gtensor import dims_except_for
    from mutopia.gtensor.interfaces import CorpusInterface
    from mutopia.utils import ParContext

    GT = model.GT
    ds = CorpusInterface(dataset)

    # A prior model's state must go, and so must the dims it left behind: the
    # pan-cancer models convolve each feature over neighbouring bins, leaving a
    # 51-long `feature` coordinate that would otherwise be handed to a model
    # expecting its own unconvolved features.  This mirrors what
    # `GtensorInterface.init_state` does before rebuilding.
    if GT.has_corpusstate(ds):
        ds.corpus = ds.corpus.drop_vars(ds.sections.groups["State"])
    for dim in ("component", "feature"):
        if dim in ds.dims:
            ds.corpus = ds.corpus.drop_dims(dim)

    # `factor_model.get_normalizers` looks its rate offset up by corpus name and
    # the model only knows the corpora it was trained on.  The offset is constant
    # per component, so it cancels under the row-normalization every consumer
    # applies -- it changes the absolute rate scale, never the topography.
    normalizers = getattr(model.factor_model_, "_normalizers", {})
    if normalizers and GT.get_name(ds) not in normalizers:
        borrowed = next(iter(normalizers))
        logger.info(
            "dataset %r is not one of the model's corpora %s; borrowing the "
            "rate normalizer from %r (constant per component, cancels under "
            "row-normalization)", GT.get_name(ds), list(normalizers), borrowed
        )
        ds.corpus = ds.corpus.assign_attrs(name=borrowed)

    # The modality predict kernels are numba functions compiled for float32
    # arguments, and `Regions/exposures` is float64 in some gTensors -- which
    # fails typing with "No matching definition for argument type(s)".  Match the
    # dtype the rest of the model's numerics already use.
    exposures = "Regions/exposures"
    if exposures in ds.corpus.data_vars and ds.corpus[exposures].dtype != np.float32:
        ds.corpus = ds.corpus.assign(
            **{exposures: ds.corpus[exposures].astype(np.float32)}
        )

    state = {}
    for sub in model.factor_model_.models.values():
        state.update(sub.prepare_corpusstate(ds))
    ds.corpus = ds.corpus.assign(**{f"State/{k}": v for k, v in state.items()})

    with ParContext(threads) as par:
        GT.update_state(ds, model.factor_model_, from_scratch=True, par_context=par)
        X = (
            model.factor_model_._get_log_mutation_rate_tensor(
                ds, par_context=par, with_context=False
            )
            .pipe(lambda X: np.exp(X - X.max(skipna=True)).fillna(0.0))
            .astype(np.float32)
        )

    rates = (
        (X * GT.get_freqs(ds)).sum(dim=dims_except_for(X.dims, "locus", "component"))
        / GT.get_regions(ds).length
    ).astype(np.float32)
    if "source" in rates.dims:
        rates = rates.sel(source=source, drop=True) if source is not None else (
            rates.isel(source=0, drop=True) if rates.sizes["source"] == 1 else rates
        )
    rates = rates.transpose("component", ...)

    out = ds.corpus if isinstance(ds, CorpusInterface) else ds
    out[key] = rates.assign_coords(component=list(model.component_names))
    logger.info('Added key: "%s" (%s)', key, dict(out[key].sizes))
    return out


def _var_names(dataset):
    """Variable names of a gTensor.

    ``mu.gt.load_dataset`` may hand back a ``CorpusInterface`` proxy, which
    forwards attributes but defines no ``__contains__`` -- so ``key in dataset``
    silently falls back to the iteration protocol and indexes with integers.
    """
    return set(getattr(dataset, "data_vars", dataset))


def _bin_key(chrom, start):
    """Join (chrom, start) into a single hashable key array."""
    return np.char.add(np.char.add(np.asarray(chrom, dtype=str), ":"),
                       np.asarray(start, dtype=np.int64).astype(str))


class TopographyUMAP(BaseEstimator):
    """Reference UMAP of component topographies, with projection of new components.

    Parameters
    ----------
    n_neighbors, min_dist, random_state, negative_sample_rate, metric, n_components
        Passed straight to :class:`umap.UMAP`.  The defaults are the parameters
        the pan-cancer reference space was built with.
    cluster_n_neighbors : int, default=10
        Neighbours used for the k-NN cluster vote.  The vote happens in the full
        feature space, matching how the reference clusters were defined.
    on_grid_mismatch : {"error", "aggregate"}, default="error"
        What to do when an incoming dataset's locus axis does not reproduce the
        reference axis.  ``"error"`` raises and explains.  ``"aggregate"`` falls
        back to a second, lossy reference space in which mesoscale-split loci are
        pooled into unique ``(chrom, start)`` bins -- see
        :meth:`_aggregate_space` for the measured cost.
    min_bin_coverage : float, default=0.9
        Under ``"aggregate"``, the minimum fraction of aggregated reference bins
        that must receive data before the projection is refused.
    outlier_quantile : float, default=0.99
        Quantile of the reference's own local-outlier-factor scores used as the
        ``is_outlier`` threshold.

    Attributes
    ----------
    reducer_ : umap.UMAP
        The fitted reducer.  ``reducer_.embedding_`` is the reference layout.
    X_ref_ : ndarray of shape (n_reference, n_bins)
    embedding_ : ndarray of shape (n_reference, n_components)
    components_ : ndarray of str
    metadata_ : DataFrame indexed by component
    bin_chrom_, bin_start_, bin_length_ : ndarray
        The reference locus axis.  Note that ``(chrom, start)`` is *not* unique:
        mesoscale features split a bin into several loci sharing coordinates.  The
        axis is matched positionally -- see :meth:`_axis_matches`.
    """

    def __init__(
        self,
        n_neighbors=5,
        min_dist=0.1,
        random_state=101,
        negative_sample_rate=3,
        metric="cosine",
        n_components=2,
        cluster_n_neighbors=10,
        on_grid_mismatch="error",
        min_bin_coverage=0.9,
        outlier_quantile=0.99,
    ):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self.negative_sample_rate = negative_sample_rate
        self.metric = metric
        self.n_components = n_components
        self.cluster_n_neighbors = cluster_n_neighbors
        self.on_grid_mismatch = on_grid_mismatch
        self.min_bin_coverage = min_bin_coverage
        self.outlier_quantile = outlier_quantile

    # ------------------------------------------------------------------ params

    @property
    def umap_params(self):
        return dict(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state,
            negative_sample_rate=self.negative_sample_rate,
            metric=self.metric,
            n_components=self.n_components,
        )

    # ---------------------------------------------------------- feature build

    @staticmethod
    def _unpack(dataset, source=None):
        """Pull the rate matrix and genomic grid out of a gTensor."""
        available = _var_names(dataset)
        if _LOCUS_KEY not in available:
            raise AttributeError(
                f"The dataset has no `{_LOCUS_KEY}`. "
                "Run `model.annot_component_distributions(data)` first."
            )

        da = dataset[_LOCUS_KEY]
        if "source" in da.dims:
            if source is not None:
                da = da.sel(source=source, drop=True)
            elif da.sizes["source"] == 1:
                da = da.isel(source=0, drop=True)
            else:
                raise ValueError(
                    f"The dataset has {da.sizes['source']} sources "
                    f"({list(da['source'].values)}); pass `source=` to pick one."
                )
        da = da.transpose("component", "locus")

        for key in ("Regions/chrom", "Regions/start", "Regions/length"):
            if key not in available:
                raise AttributeError(f"The dataset has no `{key}`; it is not a gTensor.")

        return (
            np.asarray(da.values, dtype=np.float64),
            np.asarray(dataset["Regions/chrom"].values, dtype=str),
            np.asarray(dataset["Regions/start"].values, dtype=np.int64),
            np.asarray(dataset["Regions/length"].values, dtype=np.float64),
            np.array([str(c) for c in da["component"].values]),
        )

    def _axis_matches(self, chrom, start, length):
        """Does an incoming locus axis reproduce the reference axis exactly?

        The reference axis cannot be keyed by genomic coordinate: mesoscale
        features split a bin into several loci that share ``(chrom, start, end)``
        and are not separable by any stored per-locus column -- ``(chrom, start,
        length, GeneStrand, ReplicationStrand)`` still leaves duplicate rows.  The
        axis is therefore identified *positionally*, which is sound because it is
        fully determined by the regions bed plus the mesoscale feature set, and is
        byte-identical across every cohort in the reference.
        """
        return (
            len(chrom) == len(self.bin_chrom_)
            and np.array_equal(chrom, self.bin_chrom_)
            and np.array_equal(start, self.bin_start_)
            and np.allclose(length, self.bin_length_)
        )

    @staticmethod
    def _collapse(rates, codes, length, n_out):
        """Length-weighted pooling of rate densities into ``n_out`` groups.

        ``rates`` is a density (already divided by ``Regions/length``), so a
        group's density is the length-weighted mean of its members.
        """
        acc = np.zeros((rates.shape[0], n_out), dtype=np.float64)
        weight = np.zeros(n_out, dtype=np.float64)
        np.add.at(weight, codes, length)
        np.add.at(acc.T, codes, (rates * length).T)
        covered = weight > 0
        acc[:, covered] /= weight[covered]
        return acc, covered

    def _aggregate_space(self):
        """Reference space collapsed to unique ``(chrom, start)`` bins.

        Built lazily and cached.  This is a *lossy* fallback: against the raw
        axis, pairwise cosine distances correlate r=0.89 and only 61% of
        components keep the same nearest neighbour, because the mesoscale splits
        carry real signal.  Use it only when the incoming axis cannot match.
        """
        if getattr(self, "_agg_", None) is None:
            umap = _require_umap()
            keys = _bin_key(self.bin_chrom_, self.bin_start_)
            uniq, codes = np.unique(keys, return_inverse=True)
            X, _ = self._collapse(self.X_ref_.astype(np.float64), codes, self.bin_length_, len(uniq))
            X = self._normalize(X)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*n_jobs value.*")
                reducer = umap.UMAP(**self.umap_params).fit(X)
            reducer.embedding_ = np.ascontiguousarray(self.embedding_, dtype=np.float32)

            lof, threshold = self._fit_lof(X)
            self._agg_ = dict(
                bins=uniq,
                codes=codes,
                X=X,
                reducer=reducer,
                nn=NearestNeighbors(metric="cosine").fit(X),
                lof=lof,
                threshold=threshold,
                cohort_spread=self._cohort_spread(X),
            )
        return self._agg_

    def _regrid(self, rates, chrom, start, length):
        """Align an incoming rate matrix to the reference space.

        Returns ``(X, space)`` where ``space`` is ``"strict"`` or ``"aggregate"``.
        """
        if self._axis_matches(chrom, start, length):
            return rates, "strict"

        n_ref = len(self.bin_chrom_)
        detail = (
            f"the dataset has {len(chrom)} loci on {sorted(set(chrom))}, "
            f"the reference has {n_ref} on {sorted(set(self.bin_chrom_))}"
        )
        if self.on_grid_mismatch == "error":
            raise ValueError(
                f"This dataset's locus axis does not match the reference axis ({detail}). "
                "The reference feature space is defined on a specific ordered locus axis -- "
                "the same regions bed and the same mesoscale features -- because "
                "mesoscale-split loci share genomic coordinates and cannot be matched by "
                "position in the genome. Rebuild the gTensor against the reference regions "
                "bed, or pass `on_grid_mismatch='aggregate'` to fall back to a lossy "
                "projection onto unique (chrom, start) bins."
            )
        if self.on_grid_mismatch != "aggregate":
            raise ValueError(
                f"on_grid_mismatch must be 'error' or 'aggregate', got {self.on_grid_mismatch!r}"
            )

        agg = self._aggregate_space()
        index = pd.Index(agg["bins"])
        target = index.get_indexer(_bin_key(chrom, start))
        keep = np.flatnonzero(target >= 0)
        if len(keep) == 0:
            raise ValueError(f"No genomic bin in this dataset overlaps the reference ({detail}).")

        X, covered = self._collapse(
            rates[:, keep], target[keep], length[keep], len(agg["bins"])
        )
        coverage = float(covered.mean())
        if coverage < self.min_bin_coverage:
            raise ValueError(
                f"Only {coverage:.1%} of the {len(agg['bins'])} reference bins are covered by "
                f"this dataset (minimum {self.min_bin_coverage:.0%}); {detail}."
            )
        warnings.warn(
            f"Projecting through the aggregated reference space ({coverage:.1%} of "
            f"{len(agg['bins'])} bins covered). This is lossy: collapsing mesoscale-split "
            "loci changes the geometry the reference was built on (pairwise cosine "
            "distances correlate r=0.89 with the raw axis; 61% nearest-neighbour "
            "agreement). Treat the cluster call and coordinates as indicative only.",
            stacklevel=3,
        )
        return X, "aggregate"

    @staticmethod
    def _normalize(X):
        """Row-normalize to the reference scale.

        Normalization happens *after* restriction to the reference bins, which is
        the order the reference itself was built in.
        """
        total = X.sum(axis=1, keepdims=True)
        if np.any(total <= 0):
            bad = np.flatnonzero(total.ravel() <= 0)
            raise ValueError(f"{len(bad)} component(s) have zero total rate over the reference bins.")
        return np.ascontiguousarray(X / total * _ROW_SCALE, dtype=np.float32)

    def build_features(self, dataset, source=None):
        """Turn a gTensor into a reference-aligned feature matrix.

        Returns
        -------
        X : ndarray of shape (n_components, n_bins), float32
        components : ndarray of str
        space : {"strict", "aggregate"}
            Which reference space ``X`` lives in.
        """
        check_is_fitted(self, "X_ref_")
        rates, chrom, start, length, components = self._unpack(dataset, source=source)
        X, space = self._regrid(rates, chrom, start, length)
        return self._normalize(X), components, space

    def _as_matrix(self, data, source=None):
        """Accept either a gTensor or a pre-built (X, components) pair."""
        if isinstance(data, tuple):
            X, components = data
            X = np.ascontiguousarray(X, dtype=np.float32)
            space = "strict" if X.shape[1] == self.X_ref_.shape[1] else "aggregate"
            return X, np.asarray(components, dtype=str), space
        return self.build_features(data, source=source)

    def _space(self, space):
        """The (reducer, neighbour index, reference matrix, lof, threshold) for a space."""
        if space == "strict":
            return self.reducer_, self.nn_, self.X_ref_, self.lof_, self.outlier_threshold_
        agg = self._aggregate_space()
        return agg["reducer"], agg["nn"], agg["X"], agg["lof"], agg["threshold"]

    # ------------------------------------------------------------------- fit

    def fit(self, data, metadata=None, source=None, bins=None, embedding=None):
        """Fit the reference space.

        Parameters
        ----------
        data : gTensor Dataset, or (X, components) tuple
            When a tuple is passed, ``bins`` must give the
            ``(chrom, start, length)`` locus axis its columns correspond to.
        metadata : DataFrame, optional
            Indexed by component; ``tumor_type``/``class``/``cluster_id``/
            ``cluster_name`` columns are carried into the outputs.
        embedding : ndarray, optional
            Pin the reference layout to these coordinates instead of the fitted
            ones.  Used to preserve a published layout that a current
            ``umap-learn`` no longer reproduces bit-for-bit; ``transform``
            optimizes new points against ``embedding_`` held fixed, so any fixed
            layout of the reference points defines a valid projection.
        """
        umap = _require_umap()

        if isinstance(data, tuple):
            if bins is None or len(bins) != 3:
                raise ValueError(
                    "`bins=(chrom, start, length)` is required when fitting from a matrix."
                )
            self.bin_chrom_ = np.asarray(bins[0], dtype=str)
            self.bin_start_ = np.asarray(bins[1], dtype=np.int64)
            self.bin_length_ = np.asarray(bins[2], dtype=np.float64)
            X = np.ascontiguousarray(data[0], dtype=np.float32)
            components = np.asarray(data[1], dtype=str)
        else:
            rates, chrom, start, length, components = self._unpack(data, source=source)
            self.bin_chrom_, self.bin_start_, self.bin_length_ = chrom, start, length
            X = self._normalize(rates)

        self._agg_ = None
        self.X_ref_ = X
        self.components_ = components

        meta = pd.DataFrame(index=pd.Index(components, name="component"))
        if metadata is not None:
            for field in _METADATA_FIELDS:
                if field in metadata.columns:
                    meta[field] = metadata[field].reindex(components).values
        self.metadata_ = meta

        with warnings.catch_warnings():
            # umap warns that a fixed random_state disables its parallelism.
            warnings.filterwarnings("ignore", message=".*n_jobs value.*")
            self.reducer_ = umap.UMAP(**self.umap_params).fit(X)

        if embedding is not None:
            # numba's optimize_layout kernels are typed for C-contiguous float32;
            # a pandas `.values` slice is often F-ordered and fails to type.
            embedding = np.ascontiguousarray(embedding, dtype=np.float32)
            if embedding.shape != self.reducer_.embedding_.shape:
                raise ValueError(
                    f"pinned embedding has shape {embedding.shape}, expected "
                    f"{self.reducer_.embedding_.shape}"
                )
            self.reducer_.embedding_ = embedding
        self.embedding_ = np.asarray(self.reducer_.embedding_, dtype=np.float32)

        self.nn_ = NearestNeighbors(metric="cosine").fit(X)
        self.reference_nn_distance_ = self.nn_.kneighbors(
            X, n_neighbors=2, return_distance=True
        )[0][:, 1]
        self.lof_, self.outlier_threshold_ = self._fit_lof(X)
        self.cohort_spread_ = self._cohort_spread(X)
        return self

    def _cohort_spread(self, X):
        """Within-cohort median pairwise cosine distance, across reference cohorts.

        A cohort's components should span a range of topographies.  A *set* of
        components that are all near-identical to each other has not resolved
        distinct topographic processes -- its members will each land near
        whatever reference point is closest to the centroid, and every
        per-component statistic will look reassuringly normal.  This is the only
        diagnostic that catches that, and it needs the whole set at once.
        """
        if "tumor_type" not in self.metadata_:
            return None
        from sklearn.metrics.pairwise import cosine_distances

        tumor = self.metadata_["tumor_type"].values.astype(str)
        spreads = {}
        for t in pd.unique(tumor):
            mask = tumor == t
            if mask.sum() < 3:
                continue
            D = cosine_distances(X[mask])
            spreads[t] = float(np.median(D[np.triu_indices(int(mask.sum()), k=1)]))
        return spreads or None

    @staticmethod
    def set_spread(X):
        """Median pairwise cosine distance within a set of components."""
        from sklearn.metrics.pairwise import cosine_distances

        if len(X) < 2:
            return float("nan")
        D = cosine_distances(np.asarray(X, dtype=np.float64))
        return float(np.median(D[np.triu_indices(len(X), k=1)]))

    def _check_spread(self, X, space):
        """Warn when a projected set is less topographically diverse than any cohort."""
        spreads = (self.cohort_spread_ if space == "strict"
                   else self._aggregate_space()["cohort_spread"])
        if spreads is None or len(X) < 3:
            return float("nan"), False
        observed = self.set_spread(X)
        floor = min(spreads.values())
        degenerate = observed < floor
        if degenerate:
            warnings.warn(
                f"These {len(X)} components are mutually more similar (median pairwise "
                f"cosine distance {observed:.3f}) than the components of any reference "
                f"cohort ({floor:.3f} is the lowest of {len(spreads)}). They carry little "
                "distinct topographic signal, so each will land near whatever reference "
                "point is closest to the centroid and its nearest-neighbour label will be "
                "arbitrary. Treat this projection as uninformative.",
                stacklevel=3,
            )
        return observed, degenerate

    def _fit_lof(self, X):
        """Local outlier factor over the reference, plus its flagging threshold.

        Distance to the nearest reference component is *not* a usable novelty
        score here: the features are non-negative and high-dimensional, so a bland
        profile sits near the centroid and lands closer to the reference than many
        genuine components are to each other (a uniform profile scores ~0.14 while
        the reference's own 99th-percentile neighbour distance is ~0.32).  LOF
        compares a point's local density to its neighbours' and does separate them.
        """
        lof = LocalOutlierFactor(
            n_neighbors=min(self.cluster_n_neighbors, len(X) - 1),
            metric="cosine",
            novelty=True,
        ).fit(X)
        threshold = float(np.quantile(-lof.score_samples(X), self.outlier_quantile))
        return lof, threshold

    # -------------------------------------------------------------- projection

    def transform(self, data, source=None):
        """Project components into the reference layout.

        Returns
        -------
        DataFrame indexed by component with the embedding coordinates, the
        predicted cluster, the nearest reference component and its cosine
        distance, and a local-outlier-factor novelty score with its flag.
        """
        check_is_fitted(self, "reducer_")
        X, components, space = self._as_matrix(data, source=source)
        reducer, nn, X_ref, lof, threshold = self._space(space)

        if X.shape[1] != X_ref.shape[1]:
            raise ValueError(
                f"feature matrix has {X.shape[1]} bins, the {space} reference space "
                f"has {X_ref.shape[1]}"
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*n_jobs value.*")
            coords = reducer.transform(X)
        coords = np.atleast_2d(np.asarray(coords, dtype=np.float32))

        dist, idx = nn.kneighbors(X, n_neighbors=min(self.cluster_n_neighbors, len(self.components_)))

        out = pd.DataFrame(
            coords[:, : self.n_components],
            index=pd.Index(components, name="component"),
            columns=[f"UMAP{i + 1}" for i in range(self.n_components)],
        )
        out["nn_component"] = self.components_[idx[:, 0]]
        out["nn_distance"] = dist[:, 0]
        out["outlier_score"] = -lof.score_samples(X)
        out["is_outlier"] = out["outlier_score"] > threshold
        spread, degenerate = self._check_spread(X, space)
        out["set_spread"] = spread
        out["set_is_degenerate"] = degenerate

        if "cluster_id" in self.metadata_:
            labels = self.metadata_["cluster_id"].values.astype(str)
            votes, frac = self._vote(labels, idx)
            out["cluster_id"] = votes
            out["cluster_support"] = frac
            if "cluster_name" in self.metadata_:
                names = dict(zip(labels, self.metadata_["cluster_name"].values.astype(str)))
                out["cluster_name"] = [names.get(v, "") for v in votes]
        if "class" in self.metadata_:
            out["nn_class"] = self.metadata_["class"].values.astype(str)[idx[:, 0]]

        return out

    @staticmethod
    def _vote(labels, idx):
        """Plurality vote over neighbour labels, plus the winner's support."""
        votes, support = [], []
        for row in idx:
            vals, counts = np.unique(labels[row], return_counts=True)
            winner = vals[np.argmax(counts)]
            votes.append(winner)
            support.append(counts.max() / len(row))
        return np.array(votes), np.array(support)

    def features_from_model(self, model, dataset, source=None, threads=1):
        """Reference-aligned features for a trained model, via :func:`annot_component_rates`.

        Use when the model was trained on a different regions bed than the
        reference: evaluating it on a reference-axis gTensor keeps the projection
        on the strict path.
        """
        annotated = annot_component_rates(model, dataset, source=source, threads=threads)
        return self.build_features(annotated)

    def transform_model(self, model, dataset, source=None, threads=1):
        """Project a trained model's components, evaluating it on ``dataset``'s axis.

        >>> ref = mu.load_reference_umap()
        >>> model = mu.load_model("my_cohort.model.pkl")
        >>> grid = mu.gt.load_dataset(reference_nc, with_samples=False, with_state=False)
        >>> ref.transform_model(model, grid)
        """
        X, components, _ = self.features_from_model(
            model, dataset, source=source, threads=threads
        )
        return self.transform((X, components))

    def kneighbors_model(self, model, dataset, k=10, source=None, threads=1):
        """Nearest reference components for a trained model's components."""
        X, components, _ = self.features_from_model(
            model, dataset, source=source, threads=threads
        )
        return self.kneighbors((X, components), k=k)

    def predict_cluster(self, data, source=None):
        """k-NN vote for the topography cluster, in the full feature space."""
        return self.transform(data, source=source)["cluster_id"]

    def kneighbors(self, data, k=10, source=None):
        """Nearest reference components for each input component (long form).

        This is the most robust readout -- it depends only on cosine distance in
        the original feature space, not on UMAP's approximate ``transform``.
        """
        check_is_fitted(self, "nn_")
        X, components, space = self._as_matrix(data, source=source)
        nn = self._space(space)[1]
        k = min(k, len(self.components_))
        dist, idx = nn.kneighbors(X, n_neighbors=k)

        rows = pd.DataFrame({
            "component": np.repeat(components, k),
            "rank": np.tile(np.arange(1, k + 1), len(components)),
            "ref_component": self.components_[idx.ravel()],
            "cosine_distance": dist.ravel(),
        })
        for field in _METADATA_FIELDS:
            if field in self.metadata_:
                rows[field] = self.metadata_[field].values.astype(str)[idx.ravel()]
        return rows

    # ---------------------------------------------------------- serialization

    def write_coordinates(self, path):
        """Write the coordinates-only fixture read by :func:`load_reference_coordinates`."""
        check_is_fitted(self, "embedding_")
        frame = self.metadata_.copy()
        for i in range(self.embedding_.shape[1]):
            frame[f"UMAP{i + 1}"] = self.embedding_[:, i]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        frame.to_csv(path, sep="\t")
        logger.info("wrote %s (%.1f kB)", path, os.path.getsize(path) / 1e3)
        return path

    def save(self, path, provenance=None, dtype=np.float16):
        """Write the reference space to a compressed ``.npz``.

        The fitted :class:`umap.UMAP` is deliberately *not* pickled -- it carries a
        ``pynndescent`` index with numba-typed internals whose unpickling is
        fragile across umap/numba versions.  :func:`load_reference_umap`
        reconstructs the reducer by refitting on the stored matrix, which is
        cheap at reference scale and version-robust.
        """
        check_is_fitted(self, "reducer_")
        payload = dict(
            X_ref=self.X_ref_.astype(dtype),
            embedding_ref=self.embedding_.astype(np.float32),
            bin_chrom=self.bin_chrom_,
            bin_start=self.bin_start_,
            bin_length=self.bin_length_,
            component=self.components_,
            params=json.dumps({**self.get_params()}),
            provenance=json.dumps(provenance or {}),
        )
        for field in _METADATA_FIELDS:
            if field in self.metadata_:
                payload[field] = self.metadata_[field].values.astype(str)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, **payload)
        logger.info("wrote %s (%.1f MB)", path, os.path.getsize(path) / 1e6)
        return path

    @classmethod
    def load(cls, path, verify=True, atol=1e-3):
        """Rebuild a reference space from a ``.npz`` written by :meth:`save`."""
        art = np.load(path, allow_pickle=False)
        params = json.loads(str(art["params"]))
        est = cls(**{k: v for k, v in params.items() if k in cls().get_params()})

        metadata = pd.DataFrame(
            {f: art[f] for f in _METADATA_FIELDS if f in art.files},
            index=pd.Index(art["component"], name="component"),
        )
        stored = art["embedding_ref"].astype(np.float32)
        est.fit(
            (art["X_ref"].astype(np.float32), art["component"]),
            metadata=metadata,
            bins=(art["bin_chrom"], art["bin_start"], art["bin_length"]),
            embedding=stored,
        )

        if verify:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*n_jobs value.*")
                umap = _require_umap()
                refit = umap.UMAP(**est.umap_params).fit_transform(est.X_ref_)
            if not np.allclose(refit, stored, atol=atol):
                delta = np.abs(refit - stored).max()
                warnings.warn(
                    "This umap-learn/numba build does not reproduce the stored reference "
                    f"layout (max coordinate delta {delta:.3g}); the stored layout is used. "
                    "Projections remain valid -- transform() optimizes new points against "
                    "the fixed reference layout -- but a refit would differ.",
                    stacklevel=2,
                )
        est.provenance_ = json.loads(str(art["provenance"]))
        return est


def load_reference_coordinates(path=None):
    """The canonical reference layout as a DataFrame -- coordinates only.

    The full artifact carries an 11 MB feature matrix and needs ``umap-learn`` to
    rebuild its reducer, which is a lot to pay for drawing the backdrop of a plot
    or asserting against fixed coordinates in a test.  This reads a small TSV with
    pandas alone.

    Use it to plot the reference space or to check published coordinates; use
    :func:`load_reference_umap` when you need to *project* new components.

    Returns
    -------
    DataFrame indexed by component, with ``tumor_type``, ``class``, ``cluster_id``,
    ``cluster_name``, ``UMAP1`` and ``UMAP2``.
    """
    path = path or REFERENCE_COORDINATES
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No reference coordinates at {path}. They ship with the package; if this "
            "is a source checkout, build them with "
            "`data/pancan/build_reference_artifact.py`."
        )
    return pd.read_csv(path, sep="\t", dtype={"cluster_id": str}).set_index("component")


def load_reference_umap(path=None, verify=False, download=True):
    """Load the pan-cancer topography UMAP reference.

    The ~11 MB feature matrix is a release asset, not a committed file, so the
    first call downloads it and caches it under ``~/.cache/mutopia`` (override
    with ``MUTOPIA_CACHE_DIR``).  Subsequent calls are local.  A source checkout
    that has already built the artifact in place uses that copy instead.

    If you only need the layout -- to draw it, or to check coordinates -- use
    :func:`load_reference_coordinates`, which ships in the wheel and needs
    neither the download nor ``umap-learn``.

    Parameters
    ----------
    path : str, optional
        Explicit artifact to load, bypassing discovery and download.
    verify : bool, default=False
        Refit UMAP on the reference matrix and warn if this environment does not
        reproduce the stored layout.  Off by default because it doubles load time.
    download : bool, default=True
        Whether to fetch the artifact when it is not already present.
    """
    return TopographyUMAP.load(
        _ensure_artifact(path, download=download), verify=verify
    )
