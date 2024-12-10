
from sklearn import set_config
set_config(enable_metadata_routing=True)

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, \
    QuantileTransformer, StandardScaler, RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from itertools import chain, repeat
import pandas as pd
import numpy as np
from sklearn.base import clone, OneToOneFeatureMixin
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from functools import reduce
from ..corpus_state import CorpusState as CS
from ...utils import FeatureType, logger, str_wrapped_list
from .base import get_feature_classes
from mutopia.genome_utils.fancy_iterators import streaming_groupby, \
    repeat_first, repeat_last


'''
TODO: make this work again ...
'''
def get_feature_interaction_group_idxs(
    corpus,
    transformer,
):
    None


def get_categorical_feature_idxs(
    transformer,
):  
    output_slice : slice = transformer['split_transformer']\
                .output_indices_['categorical']
    
    out_idx = list(range(output_slice.start, output_slice.stop))
    return out_idx


def get_known_categories(
    transformer,
    n_features_out,
):
    
    cat_encoder = transformer['split_transformer']\
        .named_transformers_['categorical']
    
    known_categories = [None]*n_features_out

    for idx, feature_idx in enumerate(get_categorical_feature_idxs(transformer)):
        known_categories[feature_idx] = np.arange(len(cat_encoder.categories_[idx]))

    return known_categories


def get_shared_features(*corpus_states):

    shared_features = reduce(
        lambda x,y : x.intersection(y),
        (
            set(
                (fname, FeatureType(feature.attrs['normalization']))
                for fname, feature in state.features.items() 
                if not FeatureType(feature.attrs['normalization']) in (FeatureType.MESOSCALE, FeatureType.STRAND)
            )
            for state in corpus_states
        )
    )

    seen_types = defaultdict(set)
    for fname, ftype in shared_features:
        seen_types[fname].add(ftype)
        
    for fname, ftypes in seen_types.items():
        if len(ftypes) > 1:
            raise ValueError(f'Feature {fname} has multiple normalizations: {str_wrapped_list(ftypes)}')

    return list(zip(*sorted(
        shared_features, 
        key = lambda x : not FeatureType(x[1]) == FeatureType.CATEGORICAL
    )))


class PasteTransformer(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    '''
    the `PasteTransformer` takes a dictionary of features and pastes them into a DataFrame.
    The DataFrame is necessary because the features are of heterogeneous types.
    '''

    def __init__(self,*,feature_names, add_corpus_intercepts=False):
        self.feature_names = feature_names
        self.add_corpus_intercepts = add_corpus_intercepts

    def fit(self, X, y=None,*,corpus):
        self.n_features_in_ = len(self.feature_names)
        return self
    
    def fit_transform(self, X, y=None,*,corpus):
        return self.fit(X, corpus=corpus).transform(X, corpus=corpus)
    
    def get_feature_names_out(self, input_features):
        assert hasattr(self, 'n_features_in_')
        if not self.n_features_in_ == len(input_features):
            raise ValueError('The number of input features must match the number of columns in the input data!')
        
        if self.add_corpus_intercepts:
            return ['corpus'] + list(self.feature_names)
        
        return self.feature_names
    
    def transform(self, X,*,corpus):
        missing_features = set(self.feature_names).difference( set(X.features.keys()) )
        if len(missing_features) > 0:
            raise ValueError(
                f'The following features are missing from the input data: {str_wrapped_list(missing_features)}'
            )

        df = pd.DataFrame({
            feature : X.features[feature].data
            for feature in self.feature_names
        })

        if self.add_corpus_intercepts:
            df.insert(0, 'corpus', CS.get_name(X)) 

        return df


def get_paste_transformer(feature_names, add_corpus_intercepts=False):
    return PasteTransformer(
            feature_names=feature_names, 
            add_corpus_intercepts=add_corpus_intercepts
        )\
        .set_fit_request(corpus=True)\
        .set_transform_request(corpus=True)


class Convolve(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):

    def __init__(self, width=1):
        self.width = int(width)
        if not self.width > 0:
            raise ValueError('The width must be a positive integer!')

    def fit(self, X,y=None,*,corpus):
        self.n_features_in_ = X.shape[1]
        return self
    
    def fit_transform(self, X, y = None,*,corpus):
        return self.fit(X, corpus=corpus).transform(X, corpus=corpus)

    def transform(self, X, y=None,*,corpus):

        try:
            self.n_features_in_
        except AttributeError:
            raise ValueError('This transformer must be fit before it can be used!')
        
        def collapse_group(group):
            return len(group), group[0][0]
        
        def uncollapse(idxs, group_sizes):
            return list(chain.from_iterable(repeat(i, n) for i, n in zip(idxs, group_sizes)))
                
        def r1_gt_r2(r1, r2):
            (r1_chrom, r1_start, r1_end) = r1[:3]
            (r2_chrom, r2_start, r2_end) = r2[:3]
            return r1_chrom > r2_chrom or \
                (r1_chrom == r2_chrom and r1_start >= r2_end)

        data = enumerate(zip(
            corpus.regions.chrom.data,
            corpus.regions.start.data,
            corpus.regions.end.data,
            corpus.regions.length.data,
        ))

        data = streaming_groupby(
            data,
            groupby_key=lambda _ : True, # we'll put everything into the group
            has_lapsed=lambda curr, buffer :  all(r1_gt_r2(curr[1], b[1]) for b in buffer)
        )
        data = map(lambda x : x[1], data) # remove the "group" key
        group_sizes, idxs = list(zip(*map(collapse_group, data)))

        offsets=[]
        for w in range(-self.width, 0):
            offsets.append(list(repeat_first(iter(idxs[:w]), -w)))

        offsets.append(idxs)
        
        for w in range(1, self.width+1):
            offsets.append(list(repeat_last(iter(idxs[w:]), w)))

        X = np.hstack([
            X[uncollapse(offset, group_sizes)]
            for offset in offsets
        ])

        return X


    def get_feature_names_out(self, input_features):
        assert hasattr(self, 'n_features_in_')
        if not self.n_features_in_ == len(input_features):
            raise ValueError('The number of input features must match the number of columns in the input data!')
        
        feature_names = []
        for w in range(-self.width, self.width+1):
            feature_names.extend([f'{f}:{w}' if w <= 0 else f'{f}:+{w}' for f in input_features])
        return feature_names
            

def get_convolving_transformer(width):
    return Convolve(width=width)\
        .set_fit_request(corpus=True)\
        .set_transform_request(corpus=True)


def get_categorical_transformer(
    feature_names_in,
    feature_types,
    *corpuses,
    categorical_encoder=OneHotEncoder(),
    add_corpus_intercepts=False,
):
    if not all(f == FeatureType.CATEGORICAL for f in feature_types):
        raise ValueError('All features must be categorical!')
    
    example_corpus = corpuses[0]
    categories_lists = [
        get_feature_classes(example_corpus, feature) 
        for feature in feature_names_in
    ]

    if add_corpus_intercepts:
       categories_lists = [ [CS.get_name(corpus) for corpus in corpuses] ] + categories_lists

    return categorical_encoder(categories=categories_lists)


class CPMTransformer(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.normalizers_ = np.nansum(X, axis=0, keepdims=True)
        return self
    
    def transform(self, X):
        return X/self.normalizers_*1e6

def log1p_cpm():
    return Pipeline([
        ('cpm', CPMTransformer()),
        ('log1p', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
        ('standardize', StandardScaler()),
    ])


def get_normalizing_transformer(
        feature_names_in,
        feature_types,
        *corpus_states,
    ):

    if any(f in (FeatureType.CATEGORICAL, FeatureType.MESOSCALE) for f in feature_types):
        raise ValueError('Categorical features must be processed separately!')

    type_dict = defaultdict(list)        
    for idx, feature_type in enumerate(feature_types):
        type_dict[feature_type].append(idx)

    return ColumnTransformer(
        [
            ('log1p_cpm', log1p_cpm(), type_dict[FeatureType.LOG1P_CPM]),
            ('power', log1p_cpm(), type_dict[FeatureType.POWER]),
            ('minmax', MinMaxScaler(), type_dict[FeatureType.MINMAX]),
            ('quantile', QuantileTransformer(output_distribution='uniform'), type_dict[FeatureType.QUANTILE]),
            ('standardize', StandardScaler(), type_dict[FeatureType.STANDARDIZE]),
            ('robust', RobustScaler(), type_dict[FeatureType.ROBUST]),
        ],
        verbose_feature_names_out=False,
    )


def _identity(x):
    return x

def get_feature_transformer(
    *corpuses,
    categorical_encoder=OneHotEncoder,
    convolution_width=1,
    add_corpus_intercepts=False,
    additional_transformers=[],
):
    feature_names_in, feature_types = get_shared_features(*corpuses)
    logger.info('Found locus features:{}'.format(str_wrapped_list(feature_names_in)))
    n_categorical = sum(1 for f in feature_types if f == FeatureType.CATEGORICAL)
    slice_idx = n_categorical + int(add_corpus_intercepts)

    ##
    # The corpus intercept transformer should probably be a step in the pipeline
    # which adds a new column to the feature matrix.
    ##
    pipeline_ = Pipeline([
        (
            'paste', 
            get_paste_transformer(
                feature_names=feature_names_in,
                add_corpus_intercepts=add_corpus_intercepts
            )
        ),
        (
            'split_transformer', 
            ColumnTransformer([
                (
                    'categorical', 
                    get_categorical_transformer(
                        feature_names_in[:n_categorical],
                        feature_types[:n_categorical],
                        *corpuses,
                        categorical_encoder=categorical_encoder,
                        add_corpus_intercepts=add_corpus_intercepts
                    ),
                    slice(0, slice_idx)
                ),
                (
                    'continuous',
                    Pipeline([
                        (
                            'normalize', 
                            get_normalizing_transformer(
                                feature_names_in[n_categorical:],
                                feature_types[n_categorical:],
                                *corpuses
                            )
                        ),
                        (
                            'convolve',
                            get_convolving_transformer(convolution_width) if convolution_width > 0 else \
                                FunctionTransformer(_identity, feature_names_out='one-to-one')
                        ),
                    ]),
                    slice(slice_idx, None)
                )],
                verbose_feature_names_out=False,
            )
        ),
    ])

    return (
        pipeline_,
        feature_names_in
    )
    

class StratifiedTransformer:
    '''
    Often the features for each corpus/cell type are collected in different experiements
    and so will have different distributions. The `StratifiedTransformer` fits and 
    applies a base transformer for each unique key in the `key` function. This allows
    the transformer to be fit on a per-corpus basis.
    '''
    
    def __init__(self,          
        transformer,
        key = CS.get_name,
    ):
        self.base_transformer = transformer
        self.key = key
        self._transformers = {}

    
    def transform(self, X, **kwargs):
        try:
            transformer = self._transformers[self.key(X)]
        except KeyError:
            
            transformer = clone(self.base_transformer)\
                            .fit(X,**kwargs)
            
            self._transformers[self.key(X)] = transformer

        return transformer.transform(X, **kwargs)
    

    def get_feature_names_out(self, input_features):
        return self.base_transformer.get_feature_names_out(input_features)
