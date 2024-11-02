
from abc import ABC, abstractmethod


class LocalUpdate(ABC):

    @abstractmethod
    def bound(self, gamma, *, corpus, sample, model_state):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def conditional_observation_likelihood(corpus, model_state, **sample_dict):
        raise NotImplementedError
    
    @abstractmethod
    def update_locals(self, gamma0, learning_rate, *, corpus, sample, model_state):
        raise NotImplementedError
    

    @abstractmethod
    def _get_update_fn(
        self,
        learning_rate=1.,
        subsample_rate=1.,
        *,
        corpus,
        sample,
        model_state,
    ):
        raise NotImplementedError
    

    @staticmethod
    def reduce_model_sstats(
        model,
        carry,
        corpus,
        **sample_sstats,
    ):
        return model.reduce_sparse_sstats(
            carry, 
            corpus,
            **sample_sstats
        )