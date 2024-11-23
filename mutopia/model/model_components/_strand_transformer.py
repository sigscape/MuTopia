import pandas as pd
import numpy as np
from scipy import sparse
from itertools import product
from itertools import product, repeat
from itertools import chain
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import add_dummy_feature
from ...utils import FeatureType, logger, str_wrapped_list
from .base import idx_array_to_design

def _custom_combiner(feature, category):
    return str(feature) + ":" + str(category)


class DesignMatrixHelper:

    @classmethod
    def _expand_feature_combinations(cls, n_strand_features, *categories, 
                                inverted=False):
        return list(chain(
                repeat(
                    [1,0,-1] if not inverted else [-1,0,1], 
                    n_strand_features
                ),
                categories
            ))

    @classmethod
    def onehot_encoder(cls, categories):
        def get_null(c):
            try:
                return set(c).intersection({'.','None', 0}).pop()
            except KeyError:
                return None
            
        return OneHotEncoder(
            categories=categories,
            drop=[get_null(c) for c in categories],
            sparse_output=False,
            feature_name_combiner = _custom_combiner,
        )
    
    
    @classmethod
    def get_idx_map(cls, n_strand_features, *categories, inverted=False):
        
        category_combinations = list(map(tuple, product(
            *cls._expand_feature_combinations(n_strand_features, *categories, inverted=inverted)
        )))

        label_to_idx = dict(zip(category_combinations, range(len(category_combinations))))

        return label_to_idx
    

    @classmethod
    def get_joint_encoding_matrix(cls, n_strand_features, *categories):

        categories = cls._expand_feature_combinations(n_strand_features, *categories)
        
        encoder = cls.onehot_encoder(categories)

        design = encoder.fit_transform(list(product(*categories)))
        
        return (
            design,
            encoder,
        )
    
    @classmethod
    def encoding_dim(cls, n_strand_features, *categories):
        return (
            np.prod([3]*n_strand_features + [len(c) for c in categories]),
            np.sum([2]*n_strand_features + [len(c)-1 for c in categories]),
        )
    
    @classmethod
    def one_pad_right(cls, X):
        out = sparse.hstack([
            X,
            sparse.csc_matrix(np.ones((X.shape[0],1)))
        ]).tocsr()
        out.eliminate_zeros()
        return out
    
    
    @classmethod
    def eye_pad_right(cls, X, n_repeats):
        eye_len = X.shape[0]//n_repeats
        assert eye_len*n_repeats == X.shape[0], 'X must be divisible by n_repeats'

        out = sparse.hstack([
            X,
            sparse.vstack([sparse.eye(eye_len) for _ in range(n_repeats)]).tocsr()
        ]).tocsr()

        out.eliminate_zeros()
        return out


    @classmethod
    def compose_encoding_matrix(cls, 
            base_matrix, 
            interleave_matrix, 
            shared_effects=True
        ):
        
        base_rows = base_matrix.shape[0]
        interleave_rows = interleave_matrix.shape[0]

        base_matrix_ = sparse.csr_matrix(
            base_matrix[np.repeat(np.arange(base_rows), interleave_rows), :]
        )

        shared_effects = sparse.vstack([
            sparse.csr_matrix(interleave_matrix) 
            for _ in range(base_rows)
        ]).tocsr()

        X = sparse.hstack([
            sparse.hstack([base_matrix_, shared_effects]).tocsr(),
            sparse.block_diag([interleave_matrix]*base_rows).tocsr(),
        ]).tocsr()

        X.eliminate_zeros()
        return X



class MesoscaleEncoder:

    uselog=True
    
    def fit(self, corpuses):
        
        example_features = corpuses[0].features.data_vars

        filter_features = lambda c : list([ 
            fname 
            for fname, feature in example_features.items()
            if FeatureType(feature.attrs['normalization']) in (c,)
        ])

        strand_features = filter_features(FeatureType.STRAND)
        cat_features = filter_features(FeatureType.MESOSCALE)

        for strand_feature in strand_features:
            vals = set(np.unique(example_features[strand_feature].data))
            if not vals == {-1, 0, 1}:
                raise ValueError(
                    f'The strand feature {strand_feature} has values {vals}.\n'
                    'Strand features must have values in {-1, 0, 1}.'
                )

        if self.uselog:
            logger.info('Found strand features:\n\t{}'.format(str_wrapped_list(strand_features)))
            logger.info('Found mesoscale features:\n\t{}'.format(str_wrapped_list(cat_features)))

        self.feature_names_ = strand_features + cat_features

        n_strand_features = len(strand_features)
        categories = [sorted(list(set(example_features[f].data)))[::-1] for f in cat_features]
        design_args = (n_strand_features, *categories)
        
        (self.n_states_, self.n_encoded_features_) = DesignMatrixHelper.encoding_dim(*design_args)
        self._encoding_matrix_block_, encoder = \
            DesignMatrixHelper.get_joint_encoding_matrix(*design_args) 

        self.feature_names_out_ = list(encoder.get_feature_names_out(strand_features + cat_features) )

        self.plus_encoder_map_ = DesignMatrixHelper.get_idx_map(*design_args)
        self.minus_encoder_map_ = DesignMatrixHelper.get_idx_map(*design_args, inverted=True)

        return self
    
    @property
    def encoding_matrix(self):
        return self._encoding_matrix_block_
    
    @property
    def n_coefs(self):
        return self._encoding_matrix_block_.shape[1]
    
    @property
    def n_states(self):
        return self._encoding_matrix_block_.shape[0]


    def plus_encoder_(self, x):
        return np.array([self.plus_encoder_map_[tuple(_x)] for _x in x])    
    
    def minus_encoder_(self, x):
        return np.array([self.minus_encoder_map_[tuple(_x)] for _x in x])
    
    def paste(self, corpus):
        return pd.DataFrame({
                feature_name : corpus.features[feature_name].data
                for feature_name in self.feature_names_
            }).values
    

    def encode(self, corpus, invert=False):
        mapper = self.minus_encoder_ if invert else self.plus_encoder_
        return mapper( self.paste(corpus) )
    
    
    def transform(self, corpuses, invert=False):
        return idx_array_to_design(
            self.encode(corpuses, invert=invert),
            self.n_states_
        )
    
    def independent_effects_encoding(self):
        return np.vstack([
            np.zeros((1, self.n_encoded_features_)),
            np.eye(self.n_encoded_features_)
        ])
    
    def get_feature_names_out(self):
        return self.feature_names_out_


class NormalizedMesoscaleEncoder(MesoscaleEncoder):
    
    uselog=False
    
    @classmethod
    def compose_encoding_matrix(cls, n_blocks, block_matrix):
        blocks = sparse.block_diag([block_matrix]*n_blocks)
        out = sparse.hstack([
            blocks,
            sparse.csr_matrix( np.vstack([np.eye(block_matrix.shape[0])]*n_blocks) ),
        ]).tocsr()

        out.eliminate_zeros()
        return out


    @classmethod
    def _get_intercept_mask(cls, n_blocks, n_strand_features, *categories):
        (S, R) = DesignMatrixHelper.encoding_dim(n_strand_features, *categories)
        single_block = [True] + [False]*(R-1)
        return single_block*n_blocks + [True]*S
    

    def independent_effects_encoding(self):

        s=self.n_encoded_features_ - 1

        block_matrix=np.hstack([
                np.ones(s+1)[:,None],
                np.vstack([np.zeros(s), np.eye(s)]),
            ])
        
        return self.compose_encoding_matrix(
                    self.n_blocks, 
                    block_matrix
                )[:,:-self.n_encoded_features_]
    
