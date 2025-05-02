import numpy as np
from numba import njit
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
from ..model import MuTopiaModel
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

@njit(
    "Tuple((float32, float32[:,:],float32[:]))(int64, float32[:,::1], float32[::1], float32[::1], float32[:,:], int64[:,::1])", 
    nogil=True
)
def _fast_exp_offsets(
    num_states,
    context_freqs, # (L x C*D)
    exposures, # L
    locus_effects, # L
    context_effects, # S x C
    idx_selector, # L,D
):
    
    D = 2
    L, CD = context_freqs.shape
    C = CD//D

    assert exposures.shape[0] == L
    assert locus_effects.shape[0] == L
    assert idx_selector.shape == (L, D)

    context_offsets = np.zeros((num_states, C), dtype=context_freqs.dtype)
    locus_offsets = np.zeros(L, dtype=context_freqs.dtype)
    normalizer = 0

    for l, (s_d0, s_d1) in enumerate(idx_selector):

        # (D, C)
        o = exposures[l] * context_freqs[l,:].reshape(C, D).T

        context_offsets[s_d0, :] += o[0]*locus_effects[l]
        context_offsets[s_d1, :] += o[1]*locus_effects[l]

        locus_offsets[l] += (
            (context_effects[s_d0, :] * o[0]).sum()
            + (context_effects[s_d1, :] * o[1]).sum()
        )

        normalizer += locus_offsets[l]*locus_effects[l]
        
    return (
        -np.log(normalizer),
        context_offsets.T,
        locus_offsets,
    )


contig_f32 = partial(np.ascontiguousarray, dtype=np.float32)

def _get_exp_offsets_k_c(k, corpus, num_states):

    args = (
        (
            corpus.sections.regions
            .context_frequencies
            .transpose('locus','context', 'configuration')
            .data
            .reshape(-1, 192)
        ),
        corpus.sections.regions.exposures.data,
        contig_f32( np.exp(corpus.sections.state.log_locus_distribution).sel(component=k).data ),
        np.exp(corpus.sections.state.log_context_distribution).sel(component=k).transpose(..., 'context').data,
        corpus.sections.state.mesoscale_idx.data.T,
    )

    (normalizer, context_offsets, locus_offsets) = _fast_exp_offsets(num_states, *args)

    return (normalizer, {'context_model': context_offsets, 'theta_model': locus_offsets})


class SBSModel(MuTopiaModel):

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
        empirical_bayes = True,
        begin_prior_updates = 50,
        stop_condition=50,
        # optimization settings
        test_chroms = ('chr1',),
        num_epochs = 2000,
        locus_subsample = None,
        batch_subsample = None,
        threads = 1,
        kappa = 0.5,
        tau = 1.,
        callback=None,
        eval_every=10,
        verbose=0,
        time_limit=None,
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
        self.empirical_bayes=empirical_bayes
        self.begin_prior_updates=begin_prior_updates
        self.stop_condition=stop_condition
        self.num_epochs=num_epochs
        self.locus_subsample=locus_subsample
        self.batch_subsample=batch_subsample
        self.threads=threads
        self.kappa=kappa
        self.tau=tau
        self.callback=callback
        self.eval_every=eval_every
        self.verbose=verbose
        self.time_limit=time_limit

    def sample_params(self, study, trial, extensive=0):
        return sample_params(study, trial, extensive=extensive)

    def _init_model(
        self,
        train_corpuses,
        test_corpuses,
        random_state,
        *,
        num_components,
        init_components,
        fix_components,
        seed,
        # context model
        context_reg,
        context_conditioning,
        conditioning_alpha,
        init_variance_context,
        max_iter,
        # locals model
        pi_prior,
        # locus model
        locus_model_type,
        tree_learning_rate,
        max_depth,
        max_trees_per_iter,
        max_leaf_nodes,
        min_samples_leaf,
        max_features,
        n_iter_no_change,
        use_groups,
        add_corpus_intercepts,
        convolution_width,
        l2_regularization,
        init_variance_theta,
        **kw,
    ):
        
        logger.info("Initializing model parameters and transformations...")

        context_model = StrandedContextModel(
            train_corpuses,
            DiagonalEncoder(),
            n_components=num_components,
            random_state=random_state,
            init_variance=init_variance_context,
            tol=5e-4,
            reg=context_reg,
            context_conditioning=context_conditioning,
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

        factor_model = FactorModel(
            train_corpuses,
            context_model=context_model,
            theta_model=theta_model,
            offsets_fn=partial(_get_exp_offsets_k_c, num_states=context_model.num_mesoscale_states),
        )

        return (factor_model, locals_model)



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
    def MuTopiaModel(self):
        return SBSModel


    @classmethod
    def _flatten_observations(cls, signature):
        signature = (
            super()
            ._flatten_observations(signature)
            .sel(context=COSMIC_SORT_ORDER)
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
