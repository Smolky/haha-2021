"""
    Information Gain per class
    
    This class calculates the Information Gain (Mutual Info) of a dataset
    and uses it to select the most discrimatory features
    
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser
from features.FeatureResolver import FeatureResolver
from sklearn.pipeline import Pipeline, FeatureUnion
from features.TokenizerTransformer import TokenizerTransformer


def main ():

    # var parser
    parser = DefaultParser (description = 'Merged the gold labels for HaHa test', defaults = {
        'dataset': 'haha',
        'corpus': '2021-es'
    })
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    dataset = dataset_resolver.get (args.dataset, args.corpus, 'task-1', False)
    


    # @var df Ensure if we already had the data processed
    df = dataset.get ()
    df['label'] = df['label'].astype (str)
    
    
    # @var df_gold_labels DataFrame
    df_gold_labels = pd.read_csv (dataset.get_working_dir ('dataset', 'test-with-labels.csv'))    
    df_gold_labels.index = df_gold_labels.index + 30000
    
    
    df.loc[df['__split'] == 'test', 'label'] = df_gold_labels['is_humor']
    df.loc[df['__split'] == 'test', 'humor_rating'] = df_gold_labels['humor_rating']
    df.loc[df['__split'] == 'test', 'humor_mechanism'] = df_gold_labels['humor_mechanism']
    df.loc[df['__split'] == 'test', 'humor_target'] = df_gold_labels['humor_target']
    df.loc[df['__split'] == 'test', 'votes_no'] = df_gold_labels['votes_no']
    df.loc[df['__split'] == 'test', 'votes_1'] = df_gold_labels['votes_1']
    df.loc[df['__split'] == 'test', 'votes_2'] = df_gold_labels['votes_2']
    df.loc[df['__split'] == 'test', 'votes_3'] = df_gold_labels['votes_3']
    df.loc[df['__split'] == 'test', 'votes_4'] = df_gold_labels['votes_4']
    df.loc[df['__split'] == 'test', 'votes_5'] = df_gold_labels['votes_5']
    
    
    # Reassign labels
    df.loc[df['label'] == 1, 'label'] = 'humor'
    df.loc[df['label'] == 0, 'label'] = 'non-humor'
    
    
    # Store this data on disk
    dataset.save_on_disk (df)


if __name__ == "__main__":
    main ()
