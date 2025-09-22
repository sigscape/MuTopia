from __future__ import annotations

import os
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Sequence, Tuple, List, Optional

import xarray as xr
from mutopia.palettes import categorical_palette


class ModeConfig(ABC):
    """Abstract base class for modality configuration.

    A ``ModeConfig`` defines the coordinate system, default palettes and labels,
    and a small set of utilities used by modality-specific models:

    - dimension/coordinate descriptions (``coords``, ``sizes``, ``dims``)
    - access to the modality-specific TopographyModel (``TopographyModel``)
    - helpers to load reference components and compute context frequencies
    - validation and plotting helpers for signatures/observations

    Subclasses must implement ``coords``, ``TopographyModel`` and the data
    pipeline methods ``get_context_frequencies`` and ``ingest_observations``.

    Attributes
    ----------
    MODE_ID : str
        Stable modality identifier string (e.g., "sbs").
    PALETTE : list
        Default palette for plotting signatures for this modality.
    X_LABELS : list[str]
        Optional x-axis labels to use when plotting (defaults to context labels).
    DATABASE : str
        Relative path to a JSON file with named reference components.
    """

    MODE_ID: Optional[str] = None
    PALETTE: Optional[list] = None
    X_LABELS: Optional[list[str]] = None
    DATABASE: Optional[str] = None

    @property
    def sizes(self) -> Dict[str, int]:
        """Sizes of each coordinate dimension for this modality.

        Returns
        -------
        dict[str, int]
            Mapping from dimension key to number of labels.
        """
        return {k: len(v[1]) for k, v in self.coords.items()}

    @property
    def dims(self) -> Tuple[str, ...]:
        """Tuple of dimension keys in the canonical order.

        Returns
        -------
        tuple[str, ...]
            Dimension keys corresponding to ``coords`` order.
        """
        return tuple(self.coords.keys())

    @property
    @abstractmethod
    def coords(self) -> Dict[str, Tuple[str, list]]:
        """Coordinate spec for this modality.

        Returns
        -------
        dict[str, tuple[str, list]]
            Mapping from key to a pair of (dimension name, labels).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def TopographyModel(self):
        """Modality-specific TopographyModel class.

        Implementations may import lazily to avoid cycles.
        """
        raise NotImplementedError

    @property
    def available_components(self) -> List[str]:
        """List available reference components in ``DATABASE``.

        Returns
        -------
        list[str]
            Names of components available for ``load_components``.
        """
        db_path = os.path.join(os.path.dirname(__file__), self.DATABASE)
        with open(db_path, "r") as f:
            database = json.load(f)

        return list(database.keys())

    def load_components(self, *init_components: str) -> xr.DataArray:
        """Load named reference components from the modality database.

        Parameters
        ----------
        *init_components : str
            Component names to load (must exist in ``available_components``).

        Returns
        -------
        xarray.DataArray
            Array of shape (component, context) with component spectra and
            modality metadata attached via attributes.
        """
        import numpy as np

        db_path = os.path.join(os.path.dirname(__file__), self.DATABASE)
        with open(db_path, "r") as f:
            database = json.load(f)

        comps = []
        for component in init_components:
            if not component in database:
                raise ValueError(f"Component {component} not found in database")

            comps.append([database[component][l] for l in self.coords["context"][1]])

        return xr.DataArray(
            np.array(comps, dtype=float),
            dims=("component", "context"),
            coords={
                "context": ("context", self.coords["context"][1]),
                "component": ("component", list(init_components)),
            },
            attrs={
                "dtype": self.MODE_ID,
            },
        )

    @classmethod
    @abstractmethod
    def get_context_frequencies(
        cls,
        *,
        regions_file,
        fasta_file,
    ) -> xr.DataArray:
        """Compute per-locus context frequencies for this modality.

        Implementations must return an array with (configuration, context, locus)
        or analogous dims appropriate for the modality, using dense or sparse
        storage as needed.
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
        """Ingest raw input into a canonical observation tensor.

        Implementations should return an ``xarray.DataArray`` with dims
        (configuration, context, mutation, locus) using dense or sparse
        storage. The array should include a ``name`` attribute identifying
        the sample.
        """
        raise NotImplementedError

    def _arr_to_xr(
        self,
        dim_sizes: Mapping[str, int],
        coords: Any,
        data: Any,
    ) -> xr.DataArray:
        """Helper to build a sparse DataArray from COO inputs.

        Parameters
        ----------
        dim_sizes : Mapping[str, int]
            Sizes for each modality dimension (excluding locus).
        coords : Any
            COO coordinates array or sequence.
        data : Any
            Values aligned with ``coords``.

        Returns
        -------
        xarray.DataArray
            Sparse COO-backed DataArray with dims (*sizes, locus).
        """
        from sparse import SparseArray, COO

        return xr.DataArray(
            COO(
                coords,
                data,
                shape=(*self.sizes.values(), dim_sizes["locus"]),
            ),
            dims=(*self.sizes.keys(), "locus"),
        )

    @classmethod
    def _flatten_observations(cls, signature: xr.DataArray) -> xr.DataArray:
        """Normalize signature tensor shape and ordering.

        Ensures the ``context`` dimension is the trailing dimension and
        validates minimum required dims.
        """

        cls.validate_signatures(
            signature,
            required_dims=("context",),
        )

        signature = signature.transpose(..., "context")

        return signature

    @classmethod
    def validate_signatures(
        cls,
        signature: xr.DataArray,
        required_dims: Sequence[str] = (),
    ) -> None:
        """Validate signature tensor structure for plotting or analysis.

        Parameters
        ----------
        signature : xarray.DataArray
            Signature tensor.
        required_dims : Sequence[str]
            Required dimension names.
        """

        from sparse import SparseArray

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
        signature: xr.DataArray,
        select: Sequence[str] = (),
        sig_names: Optional[Sequence[str]] = None,
    ) -> Tuple[List[Any], Sequence[str], Optional[str]]:
        """Parse a signature tensor into a list of 1D signatures.

        Returns signatures, their names, and the leading dimension name (if any).
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

            pl_signatures = list(signature.sel(genome_state=select).data)

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
        signature: xr.DataArray,
        *select: str,
        palette=categorical_palette,
        sig_names: Optional[Sequence[str]] = None,
        normalize: bool = False,
        title: Optional[str] = None,
        width: float = 5.25,
        height: float = 1.25,
        ax: Any = None,
        label_xaxis: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Plot one or more modality signatures as a linear bar plot.

        Parameters
        ----------
        signature : xarray.DataArray
            Signature tensor with a trailing ``context`` dimension. May have an
            optional leading dimension to represent multiple signatures.
        *select : str
            Optional names to select a subset of signatures from the leading
            dimension. Matching is exact or by ``name:prefix`` before the last
            colon.
        palette : sequence, optional
            Color palette for plotting multiple signatures; defaults to the
            modality palette when a single signature is plotted.
        sig_names : Sequence[str], optional
            Custom names for the selected signatures; must match selection size.
        normalize : bool, default False
            If True, normalize each signature to sum to 1 before plotting.
        title : str, optional
            Axes title.
        width, height : float, default (5.25, 1.25)
            Figure sizing passed to the underlying plotting helper.
        ax : matplotlib.axes.Axes, optional
            Existing axes to draw on; if None, a new figure/axes is created.
        label_xaxis : bool, default True
            Whether to show x-axis tick labels.
        **kwargs
            Additional keyword arguments forwarded to the plotting helper.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the rendered plot.
        """
        from mutopia.plot.signature_plot import _plot_linear_signature

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
            label_xaxis=label_xaxis,
            **dict(zip(sig_names, pl_signatures)),
        )

        if title:
            ax.set_title(title)

        return ax

    @classmethod
    def validate_observations(
        cls,
        locus_dim: int,
        observations: xr.DataArray,
    ) -> None:
        """Validate the observation tensor shape and required metadata.

        Ensures dims match (configuration, context, mutation, locus) and that a
        ``name`` attribute is present.
        """
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
