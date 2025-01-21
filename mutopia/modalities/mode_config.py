import xarray as xr
from abc import ABC, abstractmethod
from sparse import SparseArray, COO
from ..plot.signature_plot import _plot_linear_signature

class ModeConfig(ABC):

    MODE_ID = None
    PALETTE = None

    @property
    def sizes(self):
        return {
            k : len(v)
            for k,v in self.coords.items()
        }
    
    @property
    def dims(self):
        return tuple(self.coords.keys())
    
    @property
    @abstractmethod
    def coords(self) -> dict:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def make_model(self):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def sample_params(self):
        raise NotImplementedError
    
    @classmethod
    @abstractmethod
    def load_components(
        cls,
        init_components,
    ):
        raise NotImplementedError
    

    @classmethod
    @abstractmethod
    def get_context_frequencies(
        cls,
        regions_file,
        fasta_file,
        n_jobs = 1,
    ):
        raise NotImplementedError
    

    @classmethod
    @abstractmethod
    def ingest_observations(
        cls,
        *,
        input_file,
        regions_file,
        fasta_file,
        **kwargs,
    ) -> xr.DataArray:
        '''
        This should return an xr.DataArray of shape (n_configuration, n_context, n_mutation, n_locus),
        which can be either dense or sparse.

        It should have a "name" in the attributes.
        '''
        raise NotImplementedError
    

    def _arr_to_xr(self,dim_sizes, coords, data):
        return xr.DataArray(
            COO(
                coords,
                data,
                shape = (*self.sizes.values(), dim_sizes['locus']),
            ),
            dims = (*self.sizes.keys(), 'locus'),
        )
    

    @classmethod
    def _flatten_observations(cls, signature):
        
        cls.validate_signatures(
            signature,
            required_dims=('context',),
        )
        
        signature = signature.transpose(...,'context')\
            .stack(observation=('context',))\
            .assign_coords(
                observation=signature.coords['context'].values
            )
        
        return signature
    

    @classmethod
    def validate_signatures(
        cls,
        signature,
        required_dims = [],
    ):
        for dim in required_dims:
            if not dim in signature.dims:
                raise ValueError(f"Expected dimension {dim} in signature but got {signature.dims}")
            
        if not len(signature.dims) <= (len(required_dims) + 1):
            raise ValueError(
                f"Expected signature to have at most {len(required_dims) + 1} dimensions "
                f"({', '.join(required_dims)}, +1 other), but got {', '.join(signature.dims)}"
            )
        
        if isinstance(signature.data, SparseArray):
            signature.data = signature.data.todense()


    @classmethod
    def _parse_signatures(
        cls,
        signature,
        select = [],
        sig_names = None,
        normalize = False,
    ):
        '''
        Parse a signature tensor into a list of signatures.
        '''
        if len(signature.dims) > 2:
            raise ValueError('`signature` must have at most 2 dimensions.')
        
        has_extra_dims = len(signature.dims) > 1

        if len(select) > 0 and not has_extra_dims:
            raise ValueError('`select` is only valid if the signature has one extra dimension to choose from.')

        if len(select) > 0:
            lead_dim = signature.dims[0]
            if sig_names and not len(select) == len(sig_names):
                raise ValueError('If both `sig_names` and `select` are specified, they must have the same length.')
            
            pl_signatures = list(signature.loc[{lead_dim : list(select)}].data)
            sig_names = sig_names or select
            return pl_signatures, sig_names, lead_dim

        elif not has_extra_dims:
            pl_signatures = [signature.data.ravel()]
            return pl_signatures, ['Signature'], None
        
        else:
            lead_dim = signature.dims[0]
            pl_signatures = list(signature.data)
            sig_names = signature.coords[lead_dim].values
            return pl_signatures, sig_names, lead_dim
    

    @classmethod
    def plot(cls,
        signature,
        *select,
        palette = 'tab10',
        sig_names = None,
        normalize = False,
        title=None,
        **kwargs,
    ):
        
        signature = cls._flatten_observations(signature)

        pl_signatures, sig_names, legend_title = cls._parse_signatures(
            signature,
            select=select,
            sig_names=sig_names,
            normalize=normalize,
        )

        if len(pl_signatures) > 100:
            raise ValueError(
                f"Cannot plot more than 100 signatures at once. "
                f"Choose a subset of signatures to plot using the `select` argument."
            )

        if normalize:
            pl_signatures = [s/s.sum() for s in pl_signatures]
        
        ax = _plot_linear_signature(
            signature.coords['observation'].values,
            palette if len(pl_signatures) > 1 else cls.PALETTE,
            plot_kw=kwargs,
            legend_title = legend_title,
            **dict(zip(sig_names, pl_signatures)),
        )

        if title:
            ax.set_title(title)

        return ax
    

    @classmethod
    def validate_observations(
        cls,
        locus_dim,
        observations : xr.DataArray,
    ):
        if not observations.shape == (*cls.dims, locus_dim):
            raise ValueError(f"Expected tensor of shape {(*cls.dims, locus_dim)} but got {observations.shape}")
        
        if not 'name' in observations.attrs:
            raise ValueError("Expected 'name' attribute in observations")
        
        if not observations.dims == ('configuration', 'context', 'mutation', 'locus'):
            raise ValueError("Expected dimensions ('configuration', 'context', 'mutation', 'locus')")

    