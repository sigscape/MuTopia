from .modalities._sbs_nucdata import CONTEXTS
from .genome_utils.fancy_iterators import RegionOverlapComparitor
from .genome_utils.bed12_utils import unstack_regions
from .model.corpus_state import CorpusState as CS
from .utils import dims_except_for, check_structure, logger
import os
from pyfaidx import Fasta
import numpy as np

COMPLEMENT = {'A' : 'T','T' : 'A','G' : 'C','C' : 'G'}
def _revcomp(seq):
    return ''.join(reversed([COMPLEMENT[nuc] for nuc in seq]))


def deepscan_neutral_matagenesis(
    model,
    corpus,
    exposures=None,
    *,
    chrom,
    start,
    end,
    fasta,
):
    
    try:
        corpus.varm['component_distributions']
    except KeyError:
        corpus = model.annot_component_distributions(corpus)

    if exposures is None:
        try:
            contributions = corpus['contributions'].data
        except KeyError:
            raise ValueError('No exposures function provided and no contributions found in corpus')
    else:
        contributions = np.array([
            exposures(corpus, sample_name) 
            for sample_name in CS.list_samples(corpus)
        ])

    contributions/=contributions.sum(axis=1, keepdims=True)
    
    # 1. Find the regions that overlap with the specified region
    regions = zip(corpus.regions.chrom, corpus.regions.start, corpus.regions.end, range(len(corpus.regions.chrom)))
    roi=RegionOverlapComparitor(chrom, start, end)
    regions = filter(lambda region : RegionOverlapComparitor(*region) == roi, regions)
    regions_idx = [region[3] for region in regions]

    if len(regions_idx) == 0:
        raise ValueError('No regions found that overlap with the specified region')
    
    subset = corpus.isel(locus=regions_idx)

    # 2. Unstack the regions
    start, end, idx = unstack_regions(
        subset.coords['locus'].values,
        os.path.join(os.path.dirname(corpus.attrs['filename']), corpus.attrs['regions_file']),
        np.arange(len(subset.coords['locus'])),
    )

    component_dists = subset.varm['component_distributions']
    region_mutation_rate = component_dists\
        .isel(locus=idx)\
        .sum(dim=dims_except_for(component_dists.dims, 'locus','component','context'))\
        .transpose('locus', 'component', 'context')\
        .values
    
    nloci, ncomp, *_ = region_mutation_rate.shape
    region_mutation_rate = region_mutation_rate.reshape(nloci, ncomp, 32, -3).sum(axis=-1)
        
    # 3. Extract the mutation rates per sample
    positions=[]
    mutrates=[]
    nucleotides=[]

    with Fasta(fasta) as fa:
        
        for _start, _end, _mutrates in zip(start, end, region_mutation_rate):
            
            seq = fa[chrom][(_start - 1) : (_end + 2)].seq.upper()
            nucleotides.extend(seq)
            
            for pos_m1 in range(len(seq) - 3 + 1):

                context = seq[ pos_m1 : pos_m1+3 ]
                if context[1] in 'AG':
                    context = _revcomp(context)

                site_component_rates = _mutrates.T[CONTEXTS.index(context)] # 32xK -> K
                sample_mutation_rates = contributions @ site_component_rates # NxK @ K -> N

                positions.append(_start + pos_m1)
                mutrates.append(sample_mutation_rates)

    return (
        np.array(positions), 
        np.array(nucleotides), 
        np.array(mutrates)
    )


def annot_empirical_marginal(
    corpus,
):
    check_structure(corpus)
    
    X_emp = corpus.X.sum(dim=('sample',))
    X_emp = (X_emp.asdense() if X_emp.is_sparse() else X_emp)
    
    logger.info('Added key to varm: "empirical_marginal"')
    corpus.varm['empirical_marginal'] = (X_emp/corpus.regions.context_frequencies).fillna(0.)

    logger.info('Added key to varm: "empirical_locus_marginal"')
    corpus.varm['empirical_locus_marginal'] = X_emp.sum(dim=dims_except_for(X_emp.dims, 'locus'))/corpus.regions.length

    return corpus


def estimate_model_mixture(
    corpus, *models,
):
    pass

