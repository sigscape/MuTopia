from itertools import product, combinations
from functools import reduce, partial
import numpy as np
from scipy import sparse as sp
from abc import ABC, abstractmethod

class ContextEncoder(ABC):

    @abstractmethod
    def fit(self, context_names):
        pass

    @abstractmethod
    def get_feature_names_out(self):
        pass

    @property
    @abstractmethod
    def n_coefs(self):
        pass

    @property
    @abstractmethod
    def n_states(self):
        pass

    @property
    @abstractmethod
    def encoding_matrix(self):
        pass


class DiagonalEncoder(ContextEncoder):

    def fit(self, context_names):
        self.context_names = context_names
        return self
    
    def get_feature_names_out(self):
        return self.context_names
    
    @property
    def n_coefs(self):
        return len(self.context_names)
    
    @property
    def n_states(self):
        return len(self.context_names)
    
    @property
    def encoding_matrix(self):
        return sp.eye(len(self.context_names)).tocsr()



def _get_subsequence_coding(
        alphabet_by_position,
        subsequence_len, 
    ):
    def get_n_outcomes(pos):
        return reduce(
            lambda x,y: x*y,    
            (len(alphabet_by_position[j]) for j in pos)
        )

    def scan(fn, x, carry):
        out = []
        for _x in x:
            carry, _out = fn(carry, _x)
            out.append(_out)
        return carry, out
    
    observation_len = len(alphabet_by_position)

    n_coefs, subsequence_coding = scan(
        lambda carry, x : (carry + x[0], (carry, x[1])),
        map(
            lambda pos : (get_n_outcomes(pos), pos),
            combinations(range(observation_len), subsequence_len)
        ),
        0,
    )

    return n_coefs, subsequence_coding


def _make_feature_names(
    feature_name_fn,
    alphabet_by_position,
    subsequence_len, 
):
    observation_len = len(alphabet_by_position)
    
    feature_names = [
        feature_name_fn(dict(zip(pos, f)))
        for pos in combinations(range(observation_len), subsequence_len)
        for f in product(*[alphabet_by_position[j] for j in pos])
    ]

    return feature_names


def _encode_subsequences(
    alphabet_by_position, 
    subsequence_coding, 
    observation
):
    
    def _idx_of(x, a):
        return int( np.ravel_multi_index(x, [len(v) for v in a]) )
    
    return [
        start + _idx_of(
            [alphabet_by_position[i].index(observation[i]) for i in subsequence_idx],
            [alphabet_by_position[i] for i in subsequence_idx]
        )
        for start, subsequence_idx in subsequence_coding
    ]


def _offset_encoding(encoder, start, obs):
    return start + np.array(encoder(obs))

def _cat_encoding(subsequence_encoders, obs):
    return np.concatenate([enc(obs) for enc in subsequence_encoders])


def _compose_subsequence_encoders(
    alphabet_by_position, 
):
    
    subsequence_encoders = []
    n_coefs=0
    observation_len = len(alphabet_by_position)

    def _enc(encoder, start, obs):
        return start + np.array(encoder(obs))
    
    for l in range(1, observation_len+1):
        
        _n_coefs, subsequence_coding = \
            _get_subsequence_coding(alphabet_by_position, l)


        encoder = partial(
            _encode_subsequences,
            alphabet_by_position,
            subsequence_coding,
        )
        
        subsequence_encoders.append(
            partial(_offset_encoding, encoder, n_coefs)
        )

        n_coefs += _n_coefs
    
    return (
        partial(_cat_encoding, subsequence_encoders),
        n_coefs
    )


def _compose_feature_names(
    feature_name_fn,
    alphabet_by_position,
):
    observation_len = len(alphabet_by_position)
    feature_names = []
    for l in range(1, observation_len+1):
        feature_names += _make_feature_names(
            feature_name_fn,
            alphabet_by_position,
            l,
        )
    return feature_names



def _identity(x):
    return x


class KmerEncoder(ContextEncoder):

    def __init__(
        self, 
        alphabet_by_position,
        kmer_extractor = _identity,
        feature_name_fn = _identity,
    ):
        self.kmer_extractor = kmer_extractor
        self.feature_name_fn = feature_name_fn
        self.alphabet_by_position = alphabet_by_position


    def fit(self, context_names):
        self.context_names_ = context_names
        self.encoder_, self._n_coefs = _compose_subsequence_encoders(self.alphabet_by_position)
        return self
    
    def get_feature_names_out(self):
        return _compose_feature_names(
            self.feature_name_fn,
            self.alphabet_by_position
        )
    
    @property
    def n_coefs(self):
        return self._n_coefs
    
    @property
    def n_states(self):
        return len(self.context_names_)

    @property
    def encoding_matrix(self):
        
        i,j = [],[]
        for row, v in enumerate(
            map(self.kmer_extractor, self.context_names_)
        ):
            
            onehot_idx = self.encoder_(v)
            i.extend([row]*len(onehot_idx))
            j.extend(onehot_idx)
            
        X = sp.coo_matrix(
                (np.ones_like(i), (i,j)),
            ).tocsr()
        
        X.eliminate_zeros()
        return X
