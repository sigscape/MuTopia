from sklearn.preprocessing import OneHotEncoder, PowerTransformer, MinMaxScaler, \
    QuantileTransformer, StandardScaler, RobustScaler
from ..corpus_state import CorpusState as CS
from ...utils import FeatureType, logger, str_wrapped_list
from mutopia.genome_utils.fancy_iterators import streaming_groupby
from sklearn.compose import ColumnTransformer
from itertools import chain
import pandas as pd
import numpy as np
from sklearn.base import clone
from collections import defaultdict
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from functools import reduce
set_config(enable_metadata_routing=True)

'''
TODO: make this work again ...
'''
def get_feature_interaction_groups(
    corpus,
    transformer,
):
    None


def get_categorical_features(
    transformer,
):
    None


class RegionFeatureSmoother(BaseEstimator, TransformerMixin):

    def __init__(self, window_size=1000):
        self.window_size = window_size

    def fit(self, X,y=None,*,corpus):
        return self
    
    def fit_transform(self, X, y = None,*,corpus):
        return self.transform(X, corpus=corpus)

    def transform(self, X, y=None,*,corpus):

        def collapse_group(
            group,
            windowsize_threshold : int = 1000,
        ):
            '''
            Given a list of overlapping or nested regions, 
            remove any regions that are smaller than the windowsize_threshold
            by setting their window index to 0. This index is shared by the
            largest region in the group.

            The return format is a bituple where the elements are:
            (index of the region in the original list, new window index - counting from 0 within the group)
            '''
            # get the largest window by length
            get_length = lambda x : x[1][3]
            (i, _)= max( group, key = get_length )

            for j, region in group:
                if region[3] < windowsize_threshold:
                    yield (j, i)
                else:
                    yield (j, j)


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
        data = map(lambda x : x[1], data)
        data = map(list, map(collapse_group, data))
        i,j = list(zip(*chain.from_iterable(data)))

        assert (np.array(i) == np.arange(len(i))).all()

        return X.iloc[np.array(j)].copy()


def get_smoothing_transformer(*corpus_states, window_size=1000):
    return RegionFeatureSmoother(window_size=window_size)\
        .set_fit_request(corpus=True)\
        .set_transform_request(corpus=True)


class PasteTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,*,feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        missing_features = set(self.feature_names).difference( set(X.features.keys()) )
        if len(missing_features) > 0:
            raise ValueError(
                f'The following features are missing from the input data: {str_wrapped_list(missing_features)}'
            )

        return pd.DataFrame({
            f : X.features[f].data
            for f in self.feature_names
        })
    

def _get_shared_features(*corpus_states):

    shared_features = reduce(
        lambda x,y : x.intersection(y),
        (
            set(
                fname for fname, feature in state.features.items() 
                if not FeatureType(feature.attrs['normalization']) in (FeatureType.MESOSCALE, FeatureType.STRAND)
            )
            for state in corpus_states
        )
    )
    return shared_features


def get_paste_transformer(*corpus_states):
    shared_features = _get_shared_features(*corpus_states)
    logger.info('Found locus features:\n\t{}'.format(str_wrapped_list(shared_features)))
    return PasteTransformer(feature_names=shared_features)
    

def get_normalizing_transformer(
        *corpus_states,
        categorical_encoder = OneHotEncoder(sparse_output=False, drop='first'),
        add_corpus_intercepts = False,
    ):

    example_state = corpus_states[0]
    example_features = example_state.features
    
    feature_names = list(_get_shared_features(*corpus_states))

    feature_types = [
        FeatureType(example_features[feature].attrs['normalization'])
        for feature in feature_names
    ]

    type_dict = defaultdict(list)        
    for idx, feature_type in enumerate(feature_types):
        type_dict[feature_type].append(idx)

    categorical_features = PasteTransformer(
        feature_names=type_dict[FeatureType.CATEGORICAL]
    ).transform(example_state)
    
    categories_lists = [
        list(categorical_features[feature].unique()) 
        for feature in categorical_features.columns
    ]
    
    '''if self.add_corpus_intercepts:
        categories_lists = categories_lists + [[state.attrs['name'] for state in corpus_states]]
        categorical_cols = categorical_cols + [len(self.feature_names_)]'''

    cat_encoder = clone(categorical_encoder).set_params(
        categories=categories_lists
    )

    return ColumnTransformer(
        [
            ('power', PowerTransformer(), type_dict[FeatureType.POWER]),
            ('minmax', MinMaxScaler(), type_dict[FeatureType.MINMAX]),
            ('quantile', QuantileTransformer(output_distribution='uniform'), type_dict[FeatureType.QUANTILE]),
            ('standardize', StandardScaler(), type_dict[FeatureType.STANDARDIZE]),
            ('robust', RobustScaler(), type_dict[FeatureType.STANDARDIZE]),
            ('categorical', cat_encoder, type_dict[FeatureType.CATEGORICAL]),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    )


class StratifiedTransformer:
    
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
