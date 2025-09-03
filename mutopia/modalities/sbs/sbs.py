import numpy as np
from functools import reduce
from itertools import chain
from collections import Counter
from tqdm import tqdm
from pyfaidx import Fasta
import xarray as xr
from sparse import COO
import matplotlib.colors as mcolors

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

    MODE_ID = "sbs"
    MUTOPIA_TO_COSMIC_IDX = MUTOPIA_TO_COSMIC_IDX
    PALETTE = MUTATION_PALETTE
    X_LABELS = ["{}>{}".format(*m) for m in _transition_palette.keys()]
    DATABASE = "sbs/musical_sbs.json"

    @property
    def coords(self):
        return {
            "clustered": ("clustered", ["no", "yes"]),
            "configuration": ("configuration", CONFIGURATIONS),
            "context": ("context", MUTOPIA_ORDER),
            "mutation": ("context", [s[2:5] for s in MUTOPIA_ORDER]),
        }

    @property
    def TopographyModel(self):
        from .model import SBSModel
        return SBSModel

    @classmethod
    def _flatten_observations(cls, signature):
        signature = (
            super()._flatten_observations(signature).sel(context=COSMIC_SORT_ORDER)
        )

        return signature

    @classmethod
    def get_context_frequencies(
        cls,
        *,
        regions_file,
        fasta_file,
        **kw,
    ):

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

        with Fasta(fasta_file) as fasta_object:
            trinuc_matrix = [
                _count_trinucs(w, fasta_object)
                for w in tqdm(
                    stream_bed12(regions_file),
                    nrows=100,
                    desc="Aggregating trinucleotide content",
                )
            ]

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
        vcf_file,
        chr_prefix="",
        pass_only=True,
        weight_col=None,
        mutation_rate_file=None,
        sample_weight=None,
        sample_name=None,
        skip_sort=False,
        cluster=True,
        *,
        locus_dim,
        locus_coords,
        regions_file,
        fasta_file,
        **kw,
    ):
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
                shape=(2, 2, len(MUTOPIA_ORDER), locus_dim),
            ),
            dims=("clustered", "configuration", "context", "locus"),
        )

        return sample_arr.isel(locus=locus_coords)

    def ingest_uncollaposed(
        self,
        vcf_file,
        *,
        locus_dim,
        locus_coords,
        regions_file,
        fasta_file,
        **ingest_kw,
    ):

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
                shape=(2, 2, len(MUTOPIA_ORDER), locus_dim),
                has_duplicates=False,
                sorted=True,
                prune=False,
                cache=False,
            ),
            dims=("clustered", "configuration", "context", "locus"),
        )

        sample_arr = sample_arr.isel(locus=locus_coords)

        mut_id_mask = np.isin(coords[-1, :], locus_coords)
        mut_ids = [_id for _id, include in zip(mut_ids, mut_id_mask) if include]

        return (mut_ids, sample_arr)
