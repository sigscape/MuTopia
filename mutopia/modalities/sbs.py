
import json
from .base import ModeConfig
from itertools import product
import numpy as np
from ..plot.signature_plot import _plot_linear_signature
import os

COSMIC_SORT_ORDER = [
 'A[C>A]A',
 'A[C>A]C',
 'A[C>A]G',
 'A[C>A]T',
 'C[C>A]A',
 'C[C>A]C',
 'C[C>A]G',
 'C[C>A]T',
 'G[C>A]A',
 'G[C>A]C',
 'G[C>A]G',
 'G[C>A]T',
 'T[C>A]A',
 'T[C>A]C',
 'T[C>A]G',
 'T[C>A]T',
 'A[C>G]A',
 'A[C>G]C',
 'A[C>G]G',
 'A[C>G]T',
 'C[C>G]A',
 'C[C>G]C',
 'C[C>G]G',
 'C[C>G]T',
 'G[C>G]A',
 'G[C>G]C',
 'G[C>G]G',
 'G[C>G]T',
 'T[C>G]A',
 'T[C>G]C',
 'T[C>G]G',
 'T[C>G]T',
 'A[C>T]A',
 'A[C>T]C',
 'A[C>T]G',
 'A[C>T]T',
 'C[C>T]A',
 'C[C>T]C',
 'C[C>T]G',
 'C[C>T]T',
 'G[C>T]A',
 'G[C>T]C',
 'G[C>T]G',
 'G[C>T]T',
 'T[C>T]A',
 'T[C>T]C',
 'T[C>T]G',
 'T[C>T]T',
 'A[T>A]A',
 'A[T>A]C',
 'A[T>A]G',
 'A[T>A]T',
 'C[T>A]A',
 'C[T>A]C',
 'C[T>A]G',
 'C[T>A]T',
 'G[T>A]A',
 'G[T>A]C',
 'G[T>A]G',
 'G[T>A]T',
 'T[T>A]A',
 'T[T>A]C',
 'T[T>A]G',
 'T[T>A]T',
 'A[T>C]A',
 'A[T>C]C',
 'A[T>C]G',
 'A[T>C]T',
 'C[T>C]A',
 'C[T>C]C',
 'C[T>C]G',
 'C[T>C]T',
 'G[T>C]A',
 'G[T>C]C',
 'G[T>C]G',
 'G[T>C]T',
 'T[T>C]A',
 'T[T>C]C',
 'T[T>C]G',
 'T[T>C]T',
 'A[T>G]A',
 'A[T>G]C',
 'A[T>G]G',
 'A[T>G]T',
 'C[T>G]A',
 'C[T>G]C',
 'C[T>G]G',
 'C[T>G]T',
 'G[T>G]A',
 'G[T>G]C',
 'G[T>G]G',
 'G[T>G]T',
 'T[T>G]A',
 'T[T>G]C',
 'T[T>G]G',
 'T[T>G]T']

_transition_palette = {
    ('C','A') : (0.33, 0.75, 0.98),
    ('C','G') : (0.0, 0.0, 0.0),
    ('C','T') : (0.85, 0.25, 0.22),
    ('T','A') : (0.78, 0.78, 0.78),
    ('T','C') : (0.51, 0.79, 0.24),
    ('T','G') : (0.89, 0.67, 0.72)
}

MUTATION_PALETTE = [color for color in _transition_palette.values() for i in range(16)]

CONTEXTS = sorted(
                map(lambda x : ''.join(x), product('ATCG','TC','ATCG')), 
                key = lambda x : (x[1], x[0], x[2])
                )
MUTATIONS = ['T->A/C->A','T->C/C->G','T->G/C->T']
CONFIGURATIONS = ['C/T-centered','A/G-centered']

def _reformat_mut(context, mut):
    (t_centered, c_centered) = list(map(lambda s : s[3], mut.split('/')))
    mut = t_centered if context[1] == 'T' else c_centered
    cosmic = "{}[{}>{}]{}".format(context[0], context[1], mut, context[2])
    return cosmic

MUTOPIA_ORDER = [
    _reformat_mut(context, mut)
    for context in CONTEXTS
    for mut in MUTATIONS
]

MUTOPIA_TO_COSMIC_IDX = np.array([
    MUTOPIA_ORDER.index(cosmic)
    for cosmic in COSMIC_SORT_ORDER
])


class SBSMode(ModeConfig):

    MODE_ID = 'sbs'
    CONTEXTS = CONTEXTS
    MUTATIONS = MUTATIONS
    CONFIGURATIONS = CONFIGURATIONS

    @classmethod
    def load_components(cls, *init_components):
        
        filepath = os.path.join(os.path.dirname(__file__), 'musical_sbs.json')

        with open(filepath, 'r') as f:
            database = json.load(f)

        comps = []
        for component in init_components:
            
            if not component in database:
                raise ValueError(f"Component {component} not found in database")
            
            comps.append(
                np.array(
                    [database[component][context_mut] for context_mut in MUTOPIA_ORDER]
                ).reshape(
                    (cls.dim_context(), cls.dim_mutation())
                )
            )

        return np.array(comps)


    @classmethod
    def plot(cls,
        *signatures,
        palette = MUTATION_PALETTE,
        **kwargs,
    ):
        cls.validate_signatures(*signatures)
        
        _plot_linear_signature(
            COSMIC_SORT_ORDER,
            palette,
            *list(map(
                lambda s : s.ravel()[MUTOPIA_TO_COSMIC_IDX],
                signatures
            )),
            **kwargs
        )
    

    @classmethod
    def get_context_frequencies(
        cls,
        regions_file,
        fasta_file,
        n_jobs = 1,
    ):
        pass


    @classmethod
    def ingest_observations(
        cls,
        *,
        input_file,
        regions_file,
        fasta_file,
        **kwargs,
    ):
        pass