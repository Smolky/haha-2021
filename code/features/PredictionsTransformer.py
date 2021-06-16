import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle

from dlsmodels.ModelResolver import ModelResolver
from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser
from features.BertEmbeddingsTransformer import BertEmbeddingsTransformer


from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator 


class PredictionsTransformer (BaseEstimator, TransformerMixin):
    """
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
    """

    def __init__ (self, cache_file = ''):
        """
        @param model String (see Config)
        @param cache_file String
        """
        super().__init__()
        
        self.cache_file = cache_file

    
    # Return self nothing else to do here
    def fit (self, X, y = None ):
        return self     
    
    
    def transform (self, X, **transform_params):
    
        # Return tokens from cache
        if self.cache_file and os.path.exists (self.cache_file):
            return pd.read_csv (self.cache_file, header = 0, sep = ",")

        print (self.cache_file)