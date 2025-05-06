import xarray as xr
from abc import ABC, abstractmethod
from sparse import SparseArray, COO
import os
import json
import numpy as np
from ..model import TopographyModel
from ..plot.signature_plot import _plot_linear_signature
from ..utils import categorical_palette


class ModeConfig(ABC):

    MODE_ID = None
    PALETTE = None
    X_LABELS = None
    DATABASE = None

    @property
    def sizes(self):
        return {k: len(v) for k, v in self.coords.items()}

    @property
    def dims(self):
        return tuple(self.coords.keys())

    @property
    @abstractmethod
    def coords(self) -> dict:
        raise NotImplementedError

    @property
    @abstractmethod
    def TopographyModel(self) -> TopographyModel:
        raise NotImplementedError

    @classmethod
    def available_components(cls):
        filepath = os.path.join(os.path.dirname(__file__), cls.DATABASE)
        with open(filepath, "r") as f:
            database = json.load(f)

        return list(database.keys())

    @classmethod
    def load_components(cls, *init_components):

        filepath = os.path.join(os.path.dirname(__file__), cls.DATABASE)
        with open(filepath, "r") as f:
            database = json.load(f)

        comps = []
        for component in init_components:
            if not component in database:
                raise ValueError(f"Component {component} not found in database")

            comps.append([database[component][l] for l in cls().coords["context"]])

        return xr.DataArray(
            np.array(comps, dtype=float),
            dims=("component", "context"),
        )

    @classmethod
    @abstractmethod
    def get_context_frequencies(
        cls,
        *,
        regions_file,
        fasta_file,
    ):
        """
        This should return a list of context frequencies, which can be either dense or sparse.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def ingest_observations(
        cls,
        *,
        input_file,
        regions_file,
        fasta_file,
        **kwargs,
    ) -> xr.DataArray:
        """
        This should return an xr.DataArray of shape (n_configuration, n_context, n_mutation, n_locus),
        which can be either dense or sparse.

        It should have a "name" in the attributes.
        """
        raise NotImplementedError

    def _arr_to_xr(self, dim_sizes, coords, data):
        return xr.DataArray(
            COO(
                coords,
                data,
                shape=(*self.sizes.values(), dim_sizes["locus"]),
            ),
            dims=(*self.sizes.keys(), "locus"),
        )

    @classmethod
    def _flatten_observations(cls, signature):

        cls.validate_signatures(
            signature,
            required_dims=("context",),
        )

        signature = signature.transpose(..., "context")

        return signature

    @classmethod
    def validate_signatures(
        cls,
        signature,
        required_dims=[],
    ):
        for dim in required_dims:
            if not dim in signature.dims:
                raise ValueError(
                    f"Expected dimension {dim} in signature but got {signature.dims}"
                )

        if not len(signature.dims) <= (len(required_dims) + 1):
            raise ValueError(
                f"Expected signature to have at most {len(required_dims) + 1} dimensions "
                f"({', '.join(required_dims)}, +1 other), but got {', '.join(signature.dims)}"
            )

        if isinstance(signature.data, SparseArray):
            signature.data = signature.data.todense()

    @classmethod
    def _parse_signatures(
        cls,
        signature,
        select=[],
        sig_names=None,
    ):
        """
        Parse a signature tensor into a list of signatures.
        """
        if len(signature.dims) > 2:
            raise ValueError("`signature` must have at most 2 dimensions.")

        has_extra_dims = len(signature.dims) > 1

        if len(select) > 0 and not has_extra_dims:
            raise ValueError(
                "`select` is only valid if the signature has one extra dimension to choose from."
            )

        if len(select) > 0:
            lead_dim = signature.dims[0]

            select = [
                feature_name
                for feature_name in signature[lead_dim].values
                if any(
                    feature_name == q or feature_name.rsplit(":", 1)[0] == q
                    for q in select
                )
            ]

            if len(select) == 0:
                raise ValueError("No signatures selected. Check the `select` argument.")

            pl_signatures = list(signature.sel(mesoscale_state=select).data)

            if sig_names and not len(select) == len(sig_names):
                raise ValueError(
                    "If both `sig_names` and `select` are specified, they must have the same length."
                )

            sig_names = sig_names or select
            return pl_signatures, sig_names, lead_dim

        elif not has_extra_dims:
            pl_signatures = [signature.data.ravel()]
            return pl_signatures, ["Signature"], None

        else:
            lead_dim = signature.dims[0]
            pl_signatures = list(signature.data)
            sig_names = signature.coords[lead_dim].values
            return pl_signatures, sig_names, lead_dim

    @classmethod
    def plot(
        cls,
        signature,
        *select,
        palette=categorical_palette,
        sig_names=None,
        normalize=False,
        title=None,
        width=5.25,
        height=1.25,
        ax=None,
        **kwargs,
    ):

        signature = cls._flatten_observations(signature)

        pl_signatures, sig_names, legend_title = cls._parse_signatures(
            signature,
            select=select,
            sig_names=sig_names,
        )

        if len(pl_signatures) > 100:
            raise ValueError(
                f"Cannot plot more than 100 signatures at once. "
                f"Choose a subset of signatures to plot using the `select` argument."
            )

        if normalize:
            pl_signatures = [s / s.sum() for s in pl_signatures]

        ax = _plot_linear_signature(
            (
                signature.coords["context"].values
                if cls.X_LABELS is None
                else cls.X_LABELS
            ),
            palette if len(pl_signatures) > 1 else cls.PALETTE,
            plot_kw=kwargs,
            legend_title=legend_title,
            height=height,
            width=width,
            ax=ax,
            **dict(zip(sig_names, pl_signatures)),
        )

        if title:
            ax.set_title(title)

        return ax

    @classmethod
    def validate_observations(
        cls,
        locus_dim,
        observations: xr.DataArray,
    ):
        if not observations.shape == (*cls.dims, locus_dim):
            raise ValueError(
                f"Expected tensor of shape {(*cls.dims, locus_dim)} but got {observations.shape}"
            )

        if not "name" in observations.attrs:
            raise ValueError("Expected 'name' attribute in observations")

        if not observations.dims == ("configuration", "context", "mutation", "locus"):
            raise ValueError(
                "Expected dimensions ('configuration', 'context', 'mutation', 'locus')"
            )
