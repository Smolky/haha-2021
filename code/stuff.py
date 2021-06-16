"""
    To do random stuff
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import config
import argparse
import pandas as pd
import numpy as np
import pickle

from numpy import unravel_index
from tqdm import tqdm
from dlsmodels.ModelResolver import ModelResolver
from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


def main ():

    # var parser
    parser = DefaultParser (description = 'To do random stuff')
    
    
    # @var model_resolver ModelResolver
    model_resolver = ModelResolver ()

    
    # var parser
    parser.add_argument ('--model', 
        dest = 'model', 
        default = model_resolver.get_default_choice (), 
        help = 'Select the family or algorithms to evaluate', 
        choices = model_resolver.get_choices ()
    )
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    dataset = dataset_resolver.get (args.dataset, args.corpus, args.task, False)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')


    # @var df El dataframe original (en mi caso es el dataset.csv)
    df = dataset.get ()
    
    
    # Clean tweet for better matching
    df['tweet_clean'] = df['tweet_clean'].str.replace (r'\[NUMERO\]', '', regex = True);
    df['tweet_clean'] = df['tweet_clean'].str.replace (r'\[USUARIO\]', '', regex = True);
    df['tweet_clean'] = df['tweet_clean'].str.replace (r'\[HASHTAG\]\_', '', regex = True);
    df['tweet_clean'] = df['tweet_clean'].str.replace (r'(,\s*)+', ' ', regex = True);
    

    # @var satire_df DataFrame
    satire_df = df.loc[df['label'] == 'satire'].reset_index (drop = True)
    
    
    # @var non_satire_df DataFrame
    non_satire_df = df.loc[df['label'] == 'non-satire'].reset_index (drop = True)
    
    
    # @var tfidf TfidfVectorizer
    tfidf = TfidfVectorizer (
        max_features = 20000, 
        stop_words = stopwords.words ("spanish") + ['[USUARIO]', '[NUMERO]'],
        use_idf = True,
        ngram_range = (1, 1)
    )
    tfidf.fit (satire_df['tweet_clean_lowercase'])
    
    
    # @var tfidf_satire List
    tfidf_satire = tfidf.transform (satire_df['tweet_clean_lowercase'])
    
    
    # @var tfidf_non_satire List
    tfidf_non_satire = tfidf.transform (non_satire_df['tweet_clean_lowercase'])
    
    
    # @var similitudes List Cosine distance is defined as 1.0 minus the cosine similarity.
    similitudes = np.array (1 - pairwise_distances (tfidf_satire, tfidf_non_satire, metric = 'cosine'))

    
    # @var satire_texts List
    satire_texts = []


    # @var non_satire_texts List
    non_satire_texts = []
    
    
    # @var shape
    shape = similitudes.shape
    
    
    # @var total int
    total = len (satire_df)

    
    # Iterate
    for i in tqdm (range (total)):
        
        # @var maxindex
        maxindex = similitudes.argmax ()
        
        
        # @var indexes
        indexes = unravel_index (maxindex, shape)
        
        
        # Store
        satire_texts.append (indexes[0])
        non_satire_texts.append (indexes[1])
        
        # Reset, to avoid be picked again
        similitudes[indexes[0]] = 0
        similitudes[:,indexes[1]] = 0
        
        """
        if i >= 100:
            break
        """


    result = pd.concat ([
        satire_df.iloc[satire_texts]['twitter_id'].reset_index (drop = True),
        non_satire_df.iloc[non_satire_texts]['twitter_id'].reset_index (drop = True)
        ], 
        ignore_index = True, 
        axis = 1
    ).reset_index (drop = True)
    result.columns = ['satire', 'non-satire']

    result.to_csv ('benchmark.csv', index = False)
    
    
    
        
    
if __name__ == "__main__":
    main ()
