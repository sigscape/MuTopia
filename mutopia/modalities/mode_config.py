import xarray as xr
from abc import ABC, abstractmethod

class ModeConfig(ABC):

    MODE_ID = None

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


    @classmethod
    @abstractmethod
    def plot(cls,
        signatures,
        **kwargs,
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

    