
"""
xarray object layout:
/:
    dims:
        configuration
        context
        mutation
        locus
        sample
    coords:
        ...

/regions:
    chrom (locus,)
    start (locus,)
    end (locus,)
    length (locus,)
    blockstarts (locus,)
    blocklengths (locus,)
    context_frequencies (configuration, context, locus)
    exposures (locus,)

/features:
    feature_name (locus,)
        type
        group
"""

import sparse
import xarray as xr
import datatree
import numpy as np
from .reader_utils import read_windows
from ..modalities import DTYPE_MAP

WRITE_KW = dict(
        format='NETCDF4', 
        engine='netcdf4', 
    )

class BaseAccessor:
    def __init__(self, xrds):
        self._xrds = xrds

    @staticmethod
    def _validate(self, req_dim=None, req_vars=None):
        if req_dim is not None:
            if all([not dim in list(self._xrds.dims) for dim in req_dim]):
                raise AttributeError("Required dimensions are missing")
        if req_vars is not None:
            if all([not var in self._xrds.variables for var in req_vars.keys()]):
                raise AttributeError("Required variables are missing")
            

@datatree.register_datatree_accessor("modality")
class DataMode(BaseAccessor):
    def __call__(self):
        return DTYPE_MAP[self._xrds.attrs['dtype']]()


@xr.register_dataarray_accessor("is_loaded")
class IsLoaded(BaseAccessor):
    def __call__(self):
        return isinstance(self._xrds.variable._data, str)


@xr.register_dataset_accessor("coo_to_sparse")
class OnDiskCoo(BaseAccessor):

    def __call__(
        self,
        data_var='data',
    ):

        self._validate(self, ['obs_indices'], {'indices' : None, 'data' : None})

        indices = self._xrds['indices'].data.astype(int)
        data = self._xrds[data_var].data
        shape = self._xrds.attrs['shape']
        dims = tuple(list(self._xrds.coords['obs_indices'].values))
        compress = dims.index('locus')

        sparsified_data = sparse.GCXS(
            sparse.COO(
                indices,
                data,
                shape=shape,
            ),
            compressed_axes=(compress,),
        )

        return xr.DataArray(
            sparsified_data,
            dims=dims,
            attrs={k : v 
                   for k,v in self._xrds.attrs.items() 
                   if not k in ('shape', 'format')
                },
        )
    

@xr.register_dataarray_accessor("sparse_to_coo")
class OnDiskSparse(BaseAccessor):

    def __call__(
        self,
    ):

        #self._validate(self, ['configuration', 'context', 'mutation', 'locus'], None)

        sparse_matrix : sparse.GCXS = self._xrds.data

        return xr.Dataset(
            {
                'indices' : xr.DataArray(
                    sparse_matrix.tocoo().coords,
                    dims=['obs_indices', 'n_observations'],
                ),
                'data' : xr.DataArray(
                    sparse_matrix.data,
                    dims=('n_observations'),
                ),
            },
            coords={
                    "obs_indices" : list(self._xrds.dims),
                },
            attrs = {
                "shape" : sparse_matrix.shape,
                "format" : 'COO',
                **self._xrds.attrs,
            }
        )
    

def write_sample(sample_arr, filename):

    sample_name = sample_arr.attrs['name']

    dset = sample_arr.sparse_to_coo() if isinstance(sample_arr.data, sparse.SparseArray) \
                                        else sample_arr.to_dataset(name='data', promote_attrs=True)
        
    dset.to_netcdf(
            filename, 
            group=f'/_raw_samples/{sample_name}', 
            mode='a',
            **WRITE_KW
        )



def write_dataset(dataset, filename):

    dataset['/'].to_dataset().to_netcdf(filename, group='/', **WRITE_KW)

    for group in dataset.groups:
        if group == '/samples': # handled separately
            continue
        dataset[group].to_dataset()\
            .to_netcdf(filename, group=group, mode='a', **WRITE_KW)

    for sample in dataset.samples.data_vars.values():
        write_sample(sample, filename)
        


def _backend_load_sample(dataset, sample_name):

    dset = dataset._raw_samples[sample_name].to_dataset()

    if dset.attrs['format'] == 'COO':
        return dset.coo_to_sparse()
    else:
        return dset.to_dataarray().squeeze()



def load_dataset(filename):

    dataset = datatree.open_datatree(filename, cache=False)
    
    raw_sample_names = list(dataset._raw_samples.keys())

    samples_arrs = {}
    for sample_name in raw_sample_names:
        samples_arrs[sample_name] = _backend_load_sample(dataset, sample_name)
        del dataset._raw_samples[sample_name]


    datatree.DataTree(
        name='samples',
        data=xr.Dataset(
            samples_arrs,
            coords=dataset.coords
        ),
        parent=dataset,
    )

    return dataset
    

def load_old_format(corpus):

    locus_coords = ['{}:{}-{}'.format(r.chromosome, r.start, r.end) for r in corpus.regions]

    shared_coords = {
                "sample" : [sample.name for sample in corpus.samples],
                "configuration": ["C/T","A/G"],
                "context": corpus.observation_class.CONTEXTS,
                "mutation" : ["N>A","N>G","N>T/C"],
                "locus": locus_coords,
            }
    
    chrom, start, end, length = list(zip(*[
            (r.chromosome, r.start, r.end, len(r), ) for r in read_windows(corpus.regions_file)        
    ]))

    root = xr.Dataset(
        coords=shared_coords,
        attrs={
            'name' : corpus.name,
            'regions_file' : corpus.regions_file,
            'dtype' : corpus.type,
        }
    )

    regions = xr.Dataset(
        {
            'chrom' : xr.DataArray(np.array(chrom), dims=('locus')),
            'start' : xr.DataArray(np.array(start), dims=('locus')),
            'end' : xr.DataArray(np.array(end), dims=('locus')),
            'length' : xr.DataArray(np.array(length), dims=('locus')),
            "context_frequencies": xr.DataArray(
                    data=corpus.context_frequencies,
                    dims=("configuration", "context", "locus"),
                ),
            "exposures" : xr.DataArray(
                data=np.squeeze(corpus.exposures),
                dims=("locus",),
            ),
        },
        coords=shared_coords,
    )

    feature_arrays = xr.Dataset(
        {
            feature_name: xr.DataArray(
                data=feature_data['values'],
                dims=("locus",),
                attrs={
                    'type' : feature_data['type'],
                    'group' : feature_data['group'],
                }
            )
            for feature_name, feature_data in corpus.features.items()
        },
        coords=shared_coords,
    )

    sample_datasets = {}
    for sample in corpus:
        sample_datasets[sample.name] = xr.Dataset(
                    {
                        "indices" :  xr.DataArray(
                                np.array([
                                    sample.cardinality, sample.context, sample.mutation, sample.locus
                                ]),
                                dims=("obs_indices", "n_observations"),
                            ),
                        "data" : xr.DataArray(
                                data=sample.weight,
                                dims=("n_observations",),
                        )
                    },
                    coords={
                        "obs_indices" : ["configuration", "context", "mutation", "locus"],
                    },
                    attrs = {
                        "shape" : (
                            corpus.shape['cardinalities_dim'],
                            corpus.shape['context_dim'],
                            corpus.shape['mutation_dim'],
                            corpus.shape['locus_dim'],
                        ),
                        "name" : sample.name,
                    }
                )\
                .coo_to_sparse()
        

    return datatree.DataTree.from_dict({
                '/' : root,
                '/features' : feature_arrays,
                '/regions' : regions,
                '/samples' : xr.Dataset(
                    sample_datasets,    
                    coords=shared_coords
                ),
            },
        )