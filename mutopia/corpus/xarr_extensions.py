import sparse
import xarray as xr
import typing
from xarray.core.formatting_html import *
from xarray.core.formatting_html import _get_indexes_dict, _obj_repr
from collections import defaultdict



def _get_section(var_name):
        return var_name.split('/')[0].title() if '/' in var_name else 'Data'

def dataset_repr(ds) -> str:

    obj_type = f"xarray.{type(ds).__name__}"

    header_components = [f"<div class='xr-obj-type'>{escape(obj_type)}</div>"]

    sections = defaultdict(dict)
    for var_name, var in ds.data_vars.items():
        if not var_name=='X':
            sections[_get_section(var_name)][var_name] = var
        else:
            sections['X']['X'] = var

    sections = [
        dim_section(ds),
        *[
            datavar_section(
                section,
                name=section_name,
            )
            for section_name, section in sections.items()
        ],
        coord_section(ds.coords),
        index_section(_get_indexes_dict(ds.xindexes)),
        attr_section(ds.attrs),
    ]

    return _obj_repr(ds, header_components, sections)

## MONKEY PATCH!
xr.Dataset._repr_html_ = dataset_repr
xr.Dataset.features = property(lambda self : (
    self[[k for k in self.data_vars.keys() if _get_section(k)=='Features']]\
        .pipe(lambda x : x.rename({k : k.removeprefix('features/') for k in x.data_vars.keys()}))
    )
)
##


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
    

@xr.register_dataset_accessor("coo_to_sparse")
class OnDiskCoo(BaseAccessor):

    def __call__(
        self,
        data_var='data',
    ):

        self._validate(self, ['obs_indices'], {'indices' : None, 'data' : None})

        indices = self._xrds['indices'].data
        data = self._xrds[data_var].data
        shape = self._xrds.attrs['shape']
        dims = tuple(list(self._xrds.coords['obs_indices'].values))
        
        sparsified_data = sparse.COO(
            indices,
            data,
            shape=shape,
            fill_value=0.,
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

        if not isinstance(self._xrds.data, sparse.SparseArray):
            raise TypeError("Data is not in a sparse format")
        
        if not isinstance(self._xrds.data, sparse.COO):
            sparse_matrix = self._xrds.data.tocoo()
        else:
            sparse_matrix = self._xrds.data

        return xr.Dataset(
            {
                'indices' : xr.DataArray(
                    sparse_matrix.coords,
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
    
@xr.register_dataarray_accessor("ascoo")
class AsCOO(BaseAccessor):
    def __call__(self):
        if isinstance(self._xrds.data, sparse.COO):
            return self._xrds
        elif isinstance(self._xrds.data, sparse.GCXS):
            self._xrds.data = self._xrds.data.tocoo()
        else:
            raise TypeError("Data is not in a sparse format")
        
        return self._xrds
    

@xr.register_dataarray_accessor("ascsr")
class AsCSR(BaseAccessor):
    def __call__(self, *compress_dims : typing.List[str]):
        if not len(compress_dims) > 0:
            raise ValueError("Provide at least one dimension must be compressed")

        dims = self._xrds.dims
        
        self._xrds.data = sparse.GCXS(
            self._xrds.data,
            compressed_axes=tuple([
                dims.index(dim) for dim in compress_dims
            ])
        )

        return self._xrds
    

@xr.register_dataarray_accessor("asdense")
class AsCSR(BaseAccessor):
    def __call__(self):
        self._xrds.data = self._xrds.data.todense()
        return self._xrds
    
@xr.register_dataarray_accessor("is_sparse")
class IsSparse(BaseAccessor):
    def __call__(self):
        return isinstance(self._xrds.data, sparse.SparseArray)


@xr.register_dataset_accessor('fetch_sample')
class FetchSample(BaseAccessor):
    def __call__(self, sample_name):
        return self._xrds['X'].sel(sample=sample_name)


@xr.register_dataset_accessor('list_samples')
class FetchSample(BaseAccessor):
    def __call__(self):
        return self._xrds.sample.values
    

@xr.register_dataset_accessor('iter_samples')
class FetchSample(BaseAccessor):
    def __call__(self):
        for sample_name in self._xrds.list_samples():
            yield self._xrds['X'].sel(sample=sample_name)

