import sparse
import xarray as xr
import datatree
import numpy as np

WRITE_KW = dict(
        format='NETCDF4', 
        engine='netcdf4', 
    )


def _write_sample(sample_arr, filename):

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
        if group == '/layers': # handled separately
            continue

        dataset[group].to_dataset()\
            .to_netcdf(filename, group=group, mode='a', **WRITE_KW)

    for layer in dataset.layers:
        for sample_name, sample in zip(
            layer.coords['sample'],
            layer
        ):
            _write_sample(sample_name, sample, filename)
        


def _backend_load_sample(dataset, sample_name):

    dset = dataset._raw_samples[sample_name].to_dataset()

    if dset.attrs['format'] == 'COO':
        return dset.coo_to_sparse()
    else:
        return dset.to_dataarray().squeeze()


def load_dataset(filename):

    dataset = datatree.open_datatree(filename, cache=False)
    
    sample_names = list(dataset._raw_samples.keys())
    samples=[]
    for sample_name in sample_names:
        samples.append(_backend_load_sample(dataset, sample_name))
        del dataset._raw_samples[sample_name]


    sample_dims = list(samples[0].dims)
    X = xr.DataArray(
        sparse.stack(
            [s.data for s in samples],
            axis=0,
        ).change_compressed_axes(
            (0, sample_dims.index('locus')+1),
        ),
        dims=('sample', *sample_dims),
    )

    datatree.DataTree(
        name='layers',
        data=xr.Dataset(
            {'X' : X},
            coords={'sample' : sample_names},
        ),
        parent=dataset,
    )

    return dataset
    

def from_lr_corpus(corpus):

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
        )