from __future__ import annotations

import numpy as np
from functools import reduce
from itertools import chain
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from tqdm import tqdm
from pyfaidx import Fasta
import xarray as xr
from sparse import COO
import matplotlib.colors as mcolors
from mutopia.utils import logger
from mutopia.genome_utils.bed12_utils import stream_bed12
from ..mode_config import ModeConfig
from ._sbs_nucdata import *
from ._sbs_ingestion import featurize_mutations

_transition_palette = {
    ("C", "A"): mcolors.to_rgb("#427aa1ff"),
    ("C", "G"): (0.0, 0.0, 0.0),
    ("C", "T"): mcolors.to_rgb("#d1664aff"),
    ("T", "A"): (0.78, 0.78, 0.78),
    ("T", "C"): mcolors.to_rgb("#64b3aaff"),
    ("T", "G"): (0.89, 0.67, 0.72),
}

MUTATION_PALETTE = [color for color in _transition_palette.values() for i in range(16)]

MUTOPIA_TO_COSMIC_IDX = np.array(
    [MUTOPIA_ORDER.index(cosmic) for cosmic in COSMIC_SORT_ORDER]
)


class SBSMode(ModeConfig):
    """
    SBS (single base substitution) modality configuration.

    This ``ModeConfig`` defines the coordinate system, palettes, and data ingestion
    utilities used for SBS analyses. It provides helpers to:

    - build coordinate labels used across datasets (``coords``)
    - fetch the modality-specific TopographyModel class (``TopographyModel``)
    - compute reference context frequencies from genome sequence and regions
      (``get_context_frequencies``)
    - ingest observations from VCF files into the expected sparse xarray layout
      (``ingest_observations`` / ``ingest_uncollaposed``)

    Attributes
    ----------
    MODE_ID : str
        Stable modality identifier (``"sbs"``).
    MUTOPIA_TO_COSMIC_IDX : np.ndarray
        Mapping index to align Mutopia context ordering to COSMIC ordering.
    PALETTE : list[tuple[float, float, float]]
        RGB colors for the 96 SBS categories (in Mutopia order).
    X_LABELS : list[str]
        Mutation class labels (e.g., ``"C>A"``) used for display.
    DATABASE : str
        Path inside the package data for default SBS signature definitions.
    """

    MODE_ID = "sbs"
    MUTOPIA_TO_COSMIC_IDX = MUTOPIA_TO_COSMIC_IDX
    PALETTE = MUTATION_PALETTE
    X_LABELS = ["{}>{}".format(*m) for m in _transition_palette.keys()]
    DATABASE = "sbs/musical_sbs.json"

    @property
    def coords(self) -> Mapping[str, Tuple[str, List[str]]]:
        """Coordinate names and labels used by this modality.

        Returns
        -------
        Mapping[str, tuple[str, list[str]]]
            A mapping from coordinate key to a pair of (dimension name, labels).
        """
        return {
            #"clustered": ("clustered", ["no", "yes"]),
            "configuration": ("configuration", CONFIGURATIONS),
            "context": ("context", MUTOPIA_ORDER),
            "mutation": ("context", [s[2:5] for s in MUTOPIA_ORDER]),
        }

    @property
    def TopographyModel(self) -> Any:
        """Return the modality-specific TopographyModel class.

        Notes
        -----
        Imported lazily to avoid import cycles at module import time.
        """
        from .model import SBSModel
        return SBSModel

    @classmethod
    def _flatten_observations(cls, signature: xr.DataArray) -> xr.DataArray:
        """Flatten an SBS signature to the canonical layout ordered by COSMIC.

        Parameters
        ----------
        signature : xarray.DataArray
            Signature tensor with a ``context`` coordinate at least.

        Returns
        -------
        xarray.DataArray
            Reindexed DataArray with contexts ordered as ``COSMIC_SORT_ORDER``.
        """
        signature = (
            super()._flatten_observations(signature).sel(context=COSMIC_SORT_ORDER)
        )

        return signature

    @classmethod
    def get_context_frequencies(
        cls,
        *,
        regions_file: str,
        fasta_file: str,
        **kw: Any,
    ) -> xr.DataArray:
        """Compute trinucleotide context frequencies for each region.

        Parameters
        ----------
        regions_file : str
            BED12 file containing segmented regions of interest.
        fasta_file : str
            Reference FASTA file path.

        Returns
        -------
        xarray.DataArray
            Array with dims (``configuration``, ``context``, ``locus``) giving
            normalized counts for each trinucleotide context per region, with
            strand pairing applied via the two configurations.
        """

        def _get_window_seq(fasta_object, chrom, start, end):
            return fasta_object[chrom][max(start - 1, 0) : end + 1].seq.upper()

        def _rolling(seq, w=3):
            for i in range(len(seq) - w + 1):
                yield seq[i : i + w]

        def _reduce_count(counts, seq):
            if not "N" in counts:
                counts[seq] += 1
            else:
                counts["N"] += 1
            return counts

        def _count_trinucs(bed12_region, fasta_object):

            segments = map(
                lambda x: _get_window_seq(fasta_object, *x), bed12_region.segments()
            )
            trinucs = chain.from_iterable(map(_rolling, segments))
            counts = reduce(_reduce_count, trinucs, Counter())

            N_counts = counts.pop("N", 0)
            pseudocount = N_counts / (2 * len(CONTEXTS))
            contexts = CONTEXTS

            return [
                [counts[context] + pseudocount for context in contexts],
                [counts[revcomp(context)] + pseudocount for context in contexts],
            ]

        num_regions = sum(1 for _ in stream_bed12(regions_file))

        with Fasta(fasta_file) as fasta_object:
            trinuc_matrix = [
                (
                    int(w.name),
                    _count_trinucs(w, fasta_object)
                )
                for w in tqdm(
                    stream_bed12(regions_file),
                    ncols=100,
                    total=num_regions,
                    desc="Aggregating trinucleotide content",
                )
            ]

        trinuc_matrix.sort(key=lambda x: x[0])
        trinuc_matrix = [x[1] for x in trinuc_matrix]

        # LxDxC => DxCxL
        trinuc_matrix = (
            np.array(trinuc_matrix)
            .transpose(((1, 2, 0)))
            .astype(np.float32, copy=False)
        )
        # DON'T (!) add a pseudocount

        # 2xCxL => 2xCx3xL => 2x(C*3)xL
        trinuc_matrix = np.expand_dims(trinuc_matrix, axis=2)
        trinuc_matrix = np.repeat(trinuc_matrix, 3, axis=2).reshape(
            (2, -1, trinuc_matrix.shape[-1])
        )

        return xr.DataArray(
            trinuc_matrix,
            dims=("configuration", "context", "locus"),
        )

    def ingest_observations(
        self,
        vcf_file: str,
        chr_prefix: str = "",
        pass_only: bool = True,
        weight_col: str | None = None,
        mutation_rate_file: str | None = None,
        sample_weight: float | None = None,
        sample_name: str | None = None,
        skip_sort: bool = False,
        cluster: bool = True,
        *,
        locus_dim: int,
        locus_coords: Sequence[int],
        regions_file: str,
        fasta_file: str,
        **kw: Any,
    ) -> xr.DataArray:
        """Ingest a VCF into a sparse SBS observation tensor.

        Parameters
        ----------
        vcf_file : str
            Path to VCF file with somatic variants.
        chr_prefix : str, optional
            Chromosome prefix to add/remove for matching, by default "".
        pass_only : bool, optional
            Keep only PASS variants, by default True.
        weight_col : str, optional
            Optional INFO/FORMAT field to use as a weight.
        mutation_rate_file : str, optional
            Optional per-locus mutation rate file for weighting.
        sample_weight : float, optional
            Global sample weight to apply.
        sample_name : str, optional
            Override sample name.
        skip_sort : bool, optional
            Assume VCF is already sorted, by default False.
        cluster : bool, optional
            Flag to mark clustered/unclustered dimension, by default True.
        locus_dim : int
            Total number of loci across regions.
        locus_coords : Sequence[int]
            Indices mapping variants to locus positions.
        regions_file : str
            BED12 regions used to aggregate variants.
        fasta_file : str
            Reference genome FASTA file.

        Returns
        -------
        xarray.DataArray
            Sparse COO-backed array with dims (``configuration``,
            ``context``, ``locus``).
        """
        _, coords, weights = featurize_mutations(
            vcf_file,
            regions_file,
            fasta_file,
            chr_prefix=chr_prefix,
            weight_col=weight_col,
            mutation_rate_file=mutation_rate_file,
            sample_weight=sample_weight,
            sample_name=sample_name,
            pass_only=pass_only,
            skip_sort=skip_sort,
            cluster=cluster,
        )

        sample_arr = xr.DataArray(
            COO(
                coords,
                weights,
                shape=(2, len(MUTOPIA_ORDER), locus_dim),
            ),
            dims=("configuration", "context", "locus"),
        )

        if not locus_dim==len(locus_coords):
            logger.info("Subsetting to specified loci ...")
            sample_arr = sample_arr.isel(locus=locus_coords)

        return sample_arr

    def ingest_uncollaposed(
        self,
        vcf_file: str,
        *,
        locus_dim: int,
        locus_coords: Sequence[int],
        regions_file: str,
        fasta_file: str,
        **ingest_kw: Any,
    ) -> Tuple[List[str], xr.DataArray]:
        """Ingest VCF and return variant IDs alongside the observation tensor.

        This variant of ingestion keeps the per-variant identifiers, useful for
        downstream lookups or joins.

        Parameters
        ----------
        vcf_file : str
            Path to VCF file with somatic variants.
        locus_dim : int
            Total number of loci across regions.
        locus_coords : Sequence[int]
            Indices mapping variants to locus positions.
        regions_file : str
            BED12 regions used to aggregate variants.
        fasta_file : str
            Reference genome FASTA file.

        Returns
        -------
        tuple[list[str], xarray.DataArray]
            Variant identifiers and the sparse COO-backed observation tensor
            with dims (``configuration``, ``context``, ``locus``).
        """

        mut_ids, coords, weights = featurize_mutations(
            vcf_file,
            regions_file,
            fasta_file,
            **ingest_kw,
        )

        # create the sample array for inference
        sample_arr = xr.DataArray(
            COO(
                coords,
                weights,
                shape=(2, len(MUTOPIA_ORDER), locus_dim),
                has_duplicates=False,
                sorted=True,
                prune=False,
                cache=False,
            ),
            dims=("configuration", "context", "locus"),
        )

        sample_arr = sample_arr.isel(locus=locus_coords)

        mut_id_mask = np.isin(coords[-1, :], locus_coords)
        mut_ids = [_id for _id, include in zip(mut_ids, mut_id_mask) if include]

        return (mut_ids, sample_arr)
