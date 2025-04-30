import json
import numpy as np
import os
from functools import reduce, partial
from itertools import chain
from collections import Counter
from tqdm import tqdm
from pyfaidx import Fasta
import xarray as xr
from sparse import COO
import pandas as pd
import sys
from tabulate import tabulate
from ..model import *
from ..tuning import sample_params
from ..utils import logger
from ..genome_utils.bed12_utils import stream_bed12
from .mode_config import ModeConfig
from ._sbs_nucdata import *
from ._sbs_ingestion import featurize_mutations
from ._sbs_clustering import transfer_annotations_to_vcf
import matplotlib.colors as mcolors

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
    X_LABELS = ['{}>{}'.format(*m) for m in _transition_palette.keys()]
    DATABASE = "musical_sbs.json"

    @property
    def coords(self):
        return {
            "clustered": ("clustered", ["no", "yes"]),
            "configuration": ("configuration", CONFIGURATIONS),
            "context" : ("context", MUTOPIA_ORDER),
            "mutation" : ("context", [s[2:5] for s in MUTOPIA_ORDER]),
        }

    @property
    def make_model(self):
        return SBSModel

    @property
    def sample_params(self):
        return sample_params


    @classmethod
    def _flatten_observations(cls, signature):
        signature = (
            super()
            ._flatten_observations(signature)
            .sel(context=COSMIC_SORT_ORDER)
        )

        return signature


    def get_context_frequencies(
        self,
        n_jobs=1,
        *,
        regions_file,
        fasta_file,
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
            np.array(trinuc_matrix).transpose(((1, 2, 0))).astype(np.float32, copy=False)
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
                coords, weights, 
                shape=(2, 2, len(MUTOPIA_ORDER), locus_dim),
                has_duplicates=False,
                sorted=True,
                prune=False,
                cache=False,
            ),
            dims=("clustered", "configuration", "context", "locus"),
        )

        sample_arr = sample_arr.isel(locus=locus_coords)
        
        mut_id_mask = np.isin(coords[-1,:], locus_coords)
        mut_ids = [_id for _id, include in zip(mut_ids, mut_id_mask) if include]

        return (mut_ids, sample_arr)

    def annotate_mutations(
        self,
        model,
        dataset,
        vcf_file,
        chr_prefix="",
        output=sys.stdout,
        steps=5000,
        warmup=1000,
        **ingest_kw,
    ):
        
        logger.info("Ingesting mutations ...")
        mut_ids, sample_arr = self.ingest_uncollaposed(
            vcf_file,
            chr_prefix=chr_prefix,
            **ingest_kw,
        )

        logger.info("Inferring source processes ...")
        # run the inference algorithm to find the process contributions
        posterior, contributions = (
            model
            .model_state_ 
            .locals_model
            .posterior_assign_sample(
                sample_arr,
                dataset,
                model.model_state_,
                steps=steps,
                warmup=warmup,
            )
        )

        print(
            tabulate(
                pd.DataFrame(
                    contributions/contributions.sum(),
                    index=model.component_names, 
                    columns=['Fraction of\nmutations'],
                ).sort_values(
                    'Fraction of\nmutations', 
                    ascending=False
                ),
                headers='keys',
                tablefmt='simple',
                floatfmt=(".3f", ".3f")
            ), 
            file=sys.stderr
        )

        # create a dataframe to store the posterior distribution
        posterior_df = pd.DataFrame(
            np.log(posterior.T),
            index=pd.MultiIndex.from_tuples(
                mut_ids,
                names=["CHROM","POS"],
            ),
            columns=['logp_' + str(k) for k in model.component_names],
        ).reset_index()
        
        logger.info("Annotating VCF ...")
        # transfer the annotations to the VCF file
        transfer_annotations_to_vcf(
            posterior_df,
            vcf_file=vcf_file,
            chr_prefix=chr_prefix,
            description={},
            output=output,
        )


def _make_feature_name(subsequence_code):
    default = {0: "N", 1: "N", 2: "N"}
    default.update(subsequence_code)
    return "".join([default[i] for i in range(3)])


def SBSModel(
    train_corpuses,
    test_corpuses,
    num_components=15,
    init_components=[],
    fix_components=[],
    seed=0,
    # context model
    context_reg=0.0001,
    context_conditioning=1e-9,
    kmer_reg=0.005,
    conditioning_alpha=1e-9,
    context_encoder="diagonal",
    # locals model
    pi_prior=1.0,
    # locus model
    locus_model_type="gbt",
    tree_learning_rate=0.15,
    max_depth=5,
    max_trees_per_iter=25,
    max_leaf_nodes=31,
    min_samples_leaf=30,
    max_features=1.,
    n_iter_no_change=1,
    use_groups=True,
    add_corpus_intercepts=False,
    convolution_width=0,
    l2_regularization=1,
    max_iter=25,
    init_variance_theta=0.03,
    init_variance_context=0.1,
    # optimization settings
    **optimization_settings,
):

    random_state = np.random.RandomState(seed)

    logger.info("Initializing model parameters and transformations...")

    if context_encoder == "diagonal":
        kmer_encoder = DiagonalEncoder()
    elif context_encoder == "kmer":
        kmer_encoder = KmerEncoder(
            ["ACTG", "CT", "ACTG"],
            kmer_extractor=tuple,
            feature_name_fn=_make_feature_name,
        )
    else:
        raise ValueError(f"Unknown context encoder: {context_encoder}")

    context_model = StrandedContextModel(
        train_corpuses,
        kmer_encoder,
        n_components=num_components,
        random_state=random_state,
        init_variance=init_variance_context,
        tol=5e-4,
        reg=context_reg,
        context_conditioning=context_conditioning,
        kmer_reg=kmer_reg,
        conditioning_alpha=conditioning_alpha,
        init_components=init_components,
        fix_components=fix_components,
        max_iter=max_iter,
    )

    theta_model = (GBTThetaModel if locus_model_type == "gbt" else LinearThetaModel)(
        train_corpuses,
        init_variance=init_variance_theta,
        n_components=num_components,
        tree_learning_rate=tree_learning_rate,
        max_depth=max_depth,
        max_trees_per_iter=max_trees_per_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_iter_no_change=n_iter_no_change,
        use_groups=use_groups,
        random_state=random_state,
        add_corpus_intercepts=add_corpus_intercepts,
        convolution_width=convolution_width,
        l2_regularization=l2_regularization,
    )

    sparse = train_corpuses[0].X.is_sparse()
    if not all(
        corpus.X.is_sparse() == sparse for corpus in train_corpuses + test_corpuses
    ):
        raise ValueError(
            "All corpuses must be either sparse or dense - mixtures are not allowed!"
        )

    locals_model = (LDAUpdateSparse if sparse else LDAUpdateDense)(
        train_corpuses,
        n_components=num_components,
        random_state=random_state,
        prior_alpha=pi_prior,
    )

    model_state = ModelState(
        train_corpuses,
        context_model=context_model,
        theta_model=theta_model,
        locals_model=locals_model,
    )

    (model_state, train_scores, test_scores) = fit_model(
        train_corpuses,
        test_corpuses,
        model_state,
        np.random.RandomState(seed),
        **optimization_settings,
    )

    return (
        Model(model_state, train_corpuses[0].modality()),
        train_scores,
        test_scores,
    )
