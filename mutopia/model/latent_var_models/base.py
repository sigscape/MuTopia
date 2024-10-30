
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
    

class Reducer(ABC):

    @staticmethod
    @abstractmethod
    def reduce_context_sstats(context_sstats, corpus,**kw):
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def reduce_mutation_sstats(mutation_sstats, corpus,**kw):
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def reduce_theta_sstats(theta_sstats, corpus,**kw):
        raise NotImplementedError
    