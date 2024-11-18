import sparse
import xarray as xr
import typing
import datatree
from ..modalities import get_mode

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

        sparse_matrix = self._xrds.data.tocoo()

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
    

@datatree.register_datatree_accessor("modality")
class Mod(BaseAccessor):
    def __call__(self):
        return get_mode(self._xrds)