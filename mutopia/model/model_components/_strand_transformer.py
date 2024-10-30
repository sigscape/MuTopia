import pandas as pd
import numpy as np
from scipy import sparse
from itertools import product
from itertools import product, repeat
from itertools import chain
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import add_dummy_feature

from .base import idx_array_to_design

import logging
logger = logging.getLogger(' LocusRegressor')


class DesignMatrixHelper:

    @classmethod
    def _expand_feature_combinations(cls, n_strand_features, *categories, 
                                inverted=False):
        return list(chain(
                repeat(['+','.','-'] if not inverted else ['-','.','+'], n_strand_features),
                categories
            ))


    @classmethod
    def onehot_encoder(cls, categories):
        def get_null(c):
            if '.' in c:
                return '.'
            else:
                return 'None'
        
        return OneHotEncoder(
            categories=categories,
            drop=[get_null(c) for c in categories],
            sparse_output=False,
        )
    
    @classmethod
    def get_idx_map(cls, n_strand_features, *categories, inverted=False):
        
        category_combinations = list(map(tuple, product(
            *cls._expand_feature_combinations(n_strand_features, *categories, inverted=inverted)
        )))

        label_to_idx = dict(zip(category_combinations, range(len(category_combinations))))

        #return encode, category_combinations
        return label_to_idx
    

    @classmethod
    def get_joint_encoding_matrix(cls, n_strand_features, *categories):

        categories = cls._expand_feature_combinations(n_strand_features, *categories)
        
        encoder = cls.onehot_encoder(categories)

        design = encoder.fit_transform(list(product(*categories)))
        cols = encoder.get_feature_names_out()
        
        return (add_dummy_feature(design), ['dummy'] + list(cols))
    

    @classmethod
    def encoding_dim(cls, n_strand_features, *categories):
        return (
            np.prod([3]*n_strand_features + [len(c) for c in categories]),
            np.sum([2]*n_strand_features + [len(c)-1 for c in categories]) + 1,
        )
    
    
    @classmethod
    def compose_encoding_matrix(cls, n_blocks, block_matrix, shared_effects=True):
        blocks = sparse.block_diag([block_matrix]*n_blocks)
        if shared_effects:
            return sparse.hstack([
                blocks,
                sparse.csr_matrix( np.vstack([block_matrix[:,1:]]*n_blocks) ),
            ]).tocsr()
        else:
            return blocks.tocsr()


    @classmethod
    def get_intercept_mask(cls, n_blocks, n_strand_features, *categories, shared_effects=True):
        
        (_, R) = cls.encoding_dim(n_strand_features, *categories)
        
        single_block = [True] + [False]*(R-1)
        return single_block*n_blocks + ([False]*(R-1) if shared_effects else [])
    

class StrandEncoder:
    
    def __init__(self, n_blocks) -> None:
        self.n_blocks = n_blocks


    def fit(self, corpuses):
        
        example_features = corpuses[0].features.data_vars

        filter_features = lambda c : list([ 
            name for name, var in example_features.items()
            if var.attrs['type'] in (c,)
        ])

        strand_features = filter_features('cardinality')
        cat_features = filter_features('categorical')

        self.feature_names_ = strand_features + cat_features

        n_strand_features = len(strand_features)
        categories = [sorted(list(set(example_features[f].data)))[::-1] for f in cat_features]
        design_args = (n_strand_features, *categories)
        
        (self.n_states_, self.n_encoded_features_) = DesignMatrixHelper.encoding_dim(*design_args)
        self._encoding_matrix_block_, _ = \
            DesignMatrixHelper.get_joint_encoding_matrix(*design_args) 

        self.feature_names_out = ['Baseline'] + [
            f'{fname}:{cat}' 
            for fname, cats in zip(self.feature_names_, DesignMatrixHelper._expand_feature_combinations(n_strand_features, *categories)) 
            for cat in cats
            if not cat in ('.','None')
        ]

        self.plus_encoder_map_ = DesignMatrixHelper.get_idx_map(*design_args)
        self.minus_encoder_map_ = DesignMatrixHelper.get_idx_map(*design_args, inverted=True)

        self.intercept_mask_ = self._get_intercept_mask(
                                    self.n_blocks, 
                                    *design_args,
                                )

        self.encoding_matrix_ = self.compose_encoding_matrix(
            self.n_blocks, 
            self._encoding_matrix_block_,
        )

        self.encoding_matrix_.eliminate_zeros()

        return self
    
    def get_encoding_matrix(self, num_corpuses):
        out = sparse.vstack([self.encoding_matrix_]*num_corpuses).tocsr()
        out.eliminate_zeros()
        return out
    
    def get_num_coefs(self):
        return self.encoding_matrix_.shape[1]

    @classmethod
    def one_pad_right(cls, X):
        out = sparse.hstack([
            X,
            sparse.csc_matrix(np.ones((X.shape[0],1)))
        ]).tocsr()
        out.eliminate_zeros()
        return out

    @classmethod
    def compose_encoding_matrix(cls, n_blocks, block_matrix):
        blocks = sparse.block_diag([block_matrix]*n_blocks)
        out = sparse.hstack([
            blocks,
            sparse.csr_matrix( np.vstack([block_matrix[:,1:]]*n_blocks) ),
        ]).tocsr()

        out.eliminate_zeros()
        return out


    @classmethod
    def _get_intercept_mask(cls, n_blocks, n_strand_features, *categories):
        (_, R) = DesignMatrixHelper.encoding_dim(n_strand_features, *categories)
        single_block = [True] + [False]*(R-1)
        return single_block*n_blocks + [False]*(R-1)    


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

        s=self.n_encoded_features_ - 1

        block_matrix=np.hstack([
                np.ones(s+1)[:,None],
                np.vstack([np.zeros(s), np.eye(s)]),
            ])
        
        return self.compose_encoding_matrix(
                    self.n_blocks, 
                    block_matrix
                )
    


class MutationStrandEncoder(StrandEncoder):
    
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
    
