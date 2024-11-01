
import numpy as np
from ..corpus_state import CorpusState as CS
from .base import Reducer 


class SparseSuffStatReducer(Reducer):

    @staticmethod
    def reduce_context_sstats(
        context_sstats, corpus,*,
        weighted_posterior,
        locus,
        context,
        **kw,
    ):

        '''strand_states = CS.fetch_val(corpus, 'strand_idx').data\
                            [configuration, locus]

        # Use numpy advanced indexing to update context_sstats
        np.add.at(context_sstats, (slice(None), context, strand_states), weighted_posterior)

        return context_sstats'''

        strand_states = CS.fetch_val(corpus, 'strand_idx').data\
                            [locus]

        # Use numpy advanced indexing to update context_sstats
        np.add.at(context_sstats, (slice(None), context, strand_states), weighted_posterior)

        return context_sstats
    

    @staticmethod
    def reduce_mutation_sstats(
        mutation_sstats, corpus,*,
        weighted_posterior,
        configuration,
        locus,
        context,
        mutation,
        **kw,
    ):
        strand_states = CS.fetch_val(corpus, 'strand_idx').data\
                            [configuration, locus]

        # Use numpy advanced indexing to update mutation_sstats
        np.add.at(mutation_sstats, (slice(None), context, strand_states, mutation), weighted_posterior)

        return mutation_sstats
    

    @staticmethod
    def reduce_theta_sstats(
        theta_sstats, corpus,*,
        weighted_posterior,
        locus,
        **kw,
    ):
        np.add.at(theta_sstats, (slice(None), locus), weighted_posterior)

        return theta_sstats
    
    
    @staticmethod
    def reduce_locals_sstats(
        locals_sstats, corpus,*,
        gamma,
        **kw,
    ):
        locals_sstats.append(gamma)
        return locals_sstats
