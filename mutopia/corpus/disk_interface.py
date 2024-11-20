import sparse
import xarray as xr
import datatree
import numpy as np
from collections import defaultdict
from ..utils import check_corpus


WRITE_KW = dict(
        format='NETCDF4', 
        engine='netcdf4', 
    )


def _write_sample(
    sample_name,
    arr, 
    filename
):
    dset = arr.sparse_to_coo() if isinstance(arr.data, sparse.SparseArray) \
                else arr.to_dataset(name='data', promote_attrs=True)
        
    dset.to_netcdf(
            filename, 
            group=f'/raw/{sample_name}', 
            mode='a',
            **WRITE_KW
        )


def write_dataset(dataset, filename):

    check_corpus(dataset)

    dataset.to_dataset()\
        .drop_vars(dataset.data_vars)\
        .to_netcdf(
            filename, 
            group='/', 
            mode='w', 
            **WRITE_KW
        )

    for group in list(dataset.children.keys()):
        if group.startswith('/raw') or group == 'state':
            continue
        
        dataset[group].to_dataset()\
            .to_netcdf(
                filename, 
                group='/' + group, 
                mode='a', 
                encoding={
                    k : {'dtype' : v.dtype}
                    for k, v in dataset[group].data_vars.items()
                },
                **WRITE_KW
            )

    for layer_name, layer in dataset.to_dataset().data_vars.items():
        for sample_name, sample in zip(
            layer.coords['sample'].data,
            layer
        ):
            _write_sample(
                f'{layer_name}/{sample_name}',
                sample,
                filename
            )


def _backend_load_sample(dataset, sample_name):
    
    dset = dataset[sample_name].to_dataset()

    if dset.attrs['format'] == 'COO':
        return dset.coo_to_sparse().ascoo()
    else:
        return dset.to_dataarray().squeeze()


def load_dataset(filename):
    
    dataset = datatree.open_datatree(
            filename, 
            cache=False
        )

    sample_names = list(dataset['raw/X'].keys())
    layers = list(dataset.raw.keys())

    samples=defaultdict(list)
    for layer in layers:
        for sample_name in sample_names:
            samples[layer].append(
                _backend_load_sample(
                    dataset, 
                    f'raw/{layer}/{sample_name}'
                )
            )
            del dataset.raw[layer][sample_name]
        del dataset.raw[layer]
    del dataset['raw']

    samples = {
        k : xr.concat(v, dim='sample')
        for k,v in samples.items()
    }

    dataset.update(samples)

    for layer in layers:
        if isinstance(dataset[layer].data, sparse.SparseArray):
            dataset[layer].ascsr('sample','locus')
    
    dataset = dataset.load()
    return dataset
    


'''def from_lr_corpus(corpus):

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
        #coords=shared_coords,
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
        #coords={'locus' : locus_coords},
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


    def pack_samples(sample_datasets):
        sample_dims = list(next(iter(sample_datasets.values())).dims)
        return xr.DataArray(
            sparse.stack(list(
                sample_datasets.values(),
                axis=0,
            )).change_compressed_axes(
                (0, sample_dims.index('locus')+1),
            ),
            dims=('sample', *sample_dims),
        )

    return datatree.DataTree.from_dict({
                '/' : root,
                '/features' : feature_arrays,
                '/regions' : regions,
                '/layers' : xr.Dataset(
                    {'X' : pack_samples(sample_datasets)},
                ),
                '/obsm' : xr.Dataset(),
                '/varm' : xr.Dataset(),
            },
        )'''