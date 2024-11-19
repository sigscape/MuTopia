from datatree import DataTree
import xarray as xr
import numpy as np
import warnings
from typing import Union, List, Dict
from numpy.typing import ArrayLike, NDArray
from ..utils import FeatureType, check_structure
from ..modalities import get_mode
import logging
logger = logging.getLogger(' MuTensor ')
logger.setLevel(logging.INFO)


def gTensor(
    modality,
    *,
    name : str,
    chrom : List[str],
    start : List[int],
    end : List[int],
    block_starts : List[List[int]],
    block_ends : List[List[int]],
    context_frequencies : NDArray[np.number],
    exposures : NDArray[np.number],
):
    
    observation_dims = list(modality.dims.values())

    if not len(context_frequencies.shape) == len(observation_dims) + 1:
        raise ValueError(
            f'Expected `context_frequencies` to have {len(observation_dims) + 1} dimensions, '
            f'but got {len(context_frequencies.shape)} dimensions instead.'
        )
    if not observation_dims == context_frequencies.shape[:-1]:
        raise ValueError(
            f'Expected `context_frequencies` to have shape ({", ".join(observation_dims)}, N_loci), '
            f'but got {context_frequencies.shape} instead.'
        )
      
    locus_coords = [
        f'{chrom}:{start}-{end}' 
        for (chrom, start, end) in zip(chrom, start, end)
    ]

    shared_coords = {
        'locus' : locus_coords,
        **modality.coords,
    }
    obs_coords = {'sample' : []}

    region_lengths = np.sum(
        context_frequencies, 
        axis=tuple(range(context_frequencies.ndim - 1))
    )
    
    corpus = DataTree.from_dict({
            '/' : xr.Dataset(
                coords=shared_coords,
                attrs={
                    'name' : name,
                    'dtype' : modality.value,
                }
            ),
            '/regions' : xr.Dataset(
                {
                    'chrom' : xr.DataArray(np.array(chrom), dims=('locus')),
                    'start' : xr.DataArray(np.array(start), dims=('locus')),
                    'end' : xr.DataArray(np.array(end), dims=('locus')),
                    'length' : xr.DataArray(np.array(region_lengths), dims=('locus')),
                    'block_starts' : xr.DataArray(np.array(block_starts), dims=('locus')),
                    'block_ends' : xr.DataArray(np.array(block_ends), dims=('locus')),
                    'context_frequencies' : xr.DataArray(
                            data=context_frequencies,
                            dims=(*observation_dims, 'locus'),
                        ),
                    'exposures' : xr.DataArray(
                        data=np.squeeze(exposures),
                        dims=('locus',),
                    ),
                },
            ),
            '/features' : xr.Dataset(),
            '/layers' : xr.Dataset(coords=obs_coords),
            '/obsm' : xr.Dataset(coords=obs_coords),
            '/varm' : xr.Dataset(),
        },
    )

    return corpus


def add_feature(
    corpus,
    arr,
    group : str = 'default',
    *,
    name : str,
    normalization : FeatureType,
):
    
    check_structure(corpus)
    try:
        FeatureType(normalization)
    except ValueError:
        raise ValueError(
            f'Normalization type {normalization} not recognized. '
            f'Please use one of {", ".join(FeatureType.__members__)}'
        )
    
    arr = np.array(arr)
    allowed_types = FeatureType(normalization).allowed_dtypes

    if not any(np.issubdtype(arr.dtype, t) for t in allowed_types):
        raise ValueError(
            f'The feature {name} has dtype {arr.dtype} but must be one of {", ".join(map(repr, allowed_types))}.'
        )

    corpus.features[name] = xr.DataArray(
        data=np.array(arr),
        dims=('locus'),
        attrs={
            'normalization' : normalization,
            'group' : group,
        }
    )
    logger.info(f'Added key to features: "{name}"')

    return corpus



def add_sample(
    corpus,
    sample : xr.DataArray,
    *,
    name : str,
):
    check_structure(corpus)
    ## input validation
    if not isinstance(sample, xr.DataArray):
        raise ValueError('sample must be an xarray.DataArray')
    
    if 'sample' in corpus.dims:
        sample = sample.squeeze()

    required_dims = \
        set( get_mode(corpus).dims )\
        .union({'locus'})
    
    if not set(sample.dims) == required_dims:
        raise ValueError(f'sample dims must be {required_dims}')
    ##

    sample = sample\
        .ascoo()\
        .expand_dims({'sample' : [name]})\
        .ascsr('sample','locus')
    
    root = xr.concat(
        [
            corpus.to_dataset(), 
            xr.Dataset(
                {'X' : sample}, 
                coords={'sample' : [name]}
            ),
        ],
        dim='sample'
    )
    
    root.X.ascsr('sample','locus')

    corpus = DataTree(
        data=root,
        children=corpus.children
    )

    # remove and reinit the obsm section
    update_view(
        corpus,
        obsm = xr.Dataset(
            coords=corpus.coords,
        )
    )

    logger.info(f'Added sample to .X: "{name}"')

    return corpus


def update_view(
    tree,
    **kwargs : Dict[str, xr.Dataset],
):
    for node, dset in kwargs.items():
        DataTree(
            data=dset,
            name=node,
            parent=tree,
        )

    return tree


def annot_empirical_marginal(
    corpus,
):
    check_structure(corpus)
    
    with warnings.simplefilter("ignore"):
        log_em = (
            np.log( corpus.X.sum('sample') )\
                - np.log( corpus.regions.context_frequencies )
        )
    
    log_em.data = log_em.data.astype(np.float32)
    empirical_marginal = np.exp(log_em - log_em.max()).fillna(0.)
    del log_em
    
    corpus.varm['empirical_marginal'] = empirical_marginal
    logger.info('Added key to varm: "empirical_marginal"')
    return corpus



def split_by_chrom(
    corpus,
    test_chroms = ('chr1',),
):
    pass
