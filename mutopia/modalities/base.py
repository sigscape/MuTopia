import xarray as xr
import numpy as np
from abc import ABC, abstractmethod
from itertools import chain
import matplotlib.pyplot as plt


class ModeConfig(ABC):

    CONTEXTS = None
    MUTATIONS = None
    CONFIGURATIONS = None
    MODE_ID = None

    @classmethod
    @abstractmethod
    def dims(cls):
        return {
            'configuration' : len(cls.CONFIGURATIONS),
            'context' : len(cls.CONTEXTS),
            'mutation' : len(cls.MUTATIONS),
        }
    
    @classmethod
    @abstractmethod
    def coords(cls):
        return {
            'configuration' : cls.CONFIGURATIONS,
            'context' : cls.CONTEXTS,
            'mutation' : cls.MUTATIONS,
        }

    
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
        *signatures,
    ):
        for sig in signatures:
            if not sig.shape == cls.dims()[1:]:
                raise ValueError(f"Expected tensor of shape {cls.dims()[1:]} but got {sig.shape}")


    @classmethod
    @abstractmethod
    def plot(cls,
        *signatures,
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

    