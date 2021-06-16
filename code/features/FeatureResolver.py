import config

from .SentenceEmbeddingsTransformer import SentenceEmbeddingsTransformer
from .BertEmbeddingsTransformer import BertEmbeddingsTransformer
from .LinguisticFeaturesTransformer import LinguisticFeaturesTransformer
from .TokenizerTransformer import TokenizerTransformer
from .NegationTransformer import NegationTransformer
from .ContextualFeaturesTransformer import ContextualFeaturesTransformer
from .PredictionsTransformer import PredictionsTransformer


class FeatureResolver ():
    """
    FeatureResolver
    
    Determines which feature set should I load
    """

    def __init__ (self, dataset):
        """
        @param dataset DatasetBase
        """
        self.dataset = dataset
        
        
    def get_suggested_cache_file (self, features, task_type = 'classification'):
        """
        Returns the suggested cache file for each feature set. The interesting point 
        here is that each feature set could have a better feature selection technique
        
        @param features String
        @param task_type String
        
        @todo It will be interesting that the features will be part of the hyperparameter 
              tunning
        
        @return String
        """
        
        if 'lf' == features:
            return 'lf_minmax_ig.csv' if task_type in ['multi_label', 'classification'] else 'lf_minmax_regression.csv'

        if 'se' == features:
            return 'se_ig.csv' if task_type in ['multi_label', 'classification'] else 'se_regression.csv'

        if 'be' == features:
            return 'be_ig.csv' if task_type in ['multi_label', 'classification'] else 'be_regression.csv'
            
        if 'bf' == features:
            return 'bf.csv'
    
        if 'we' == features:
            return 'we.csv'
            
        if 'ne' == features:
            return 'ne_robust.csv'
            
        if 'cf' == features:
            return 'cf_robust.csv'

        if 'pr' == features:
            return 'pr.csv'
            

    def get (self, features, cache_file):
    
        # @var fasttext_model String
        fasttext_model = config.pretrained_models[self.dataset.get_dataset_language ()]['fasttext']['binary']
        
        
        if 'lf' == features:
            return LinguisticFeaturesTransformer (cache_file = cache_file)

        if 'se' == features:
            return SentenceEmbeddingsTransformer (fasttext_model, cache_file = cache_file, field = 'tweet_clean')

        if 'be' == features:
        
            # @var huggingface_model String
            # @todo. Fix this
            huggingface_model = 'dccuchile/bert-base-spanish-wwm-uncased'
        
            return BertEmbeddingsTransformer (huggingface_model, cache_file = cache_file, field = 'tweet_clean')

        if 'bf' == features:
        
            # @var huggingface_model String
            # @todo. Fix this
            huggingface_model = ''
            
            
            return BertEmbeddingsTransformer (huggingface_model, cache_file = cache_file, field = 'tweet_clean')

        if 'we' == features:
            return TokenizerTransformer (cache_file = cache_file, field = 'tweet_clean')
            
        if 'ne' == features:
            return NegationTransformer (cache_file = cache_file)

        if 'cf' == features:
            return ContextualFeaturesTransformer (cache_file = cache_file)

        if 'pr' == features:
            return PredictionsTransformer (cache_file = cache_file)
            