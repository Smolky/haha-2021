"""
    HAHA'2021 submission
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys

import os
import sys
import argparse
import pandas as pd
import numpy as np
import csv
import sklearn
import itertools

from pathlib import Path

from dlsdatasets.DatasetResolver import DatasetResolver
from dlsmodels.ModelResolver import ModelResolver
from features.FeatureResolver import FeatureResolver
from utils.Parser import DefaultParser
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error
from scipy.special import softmax
from utils.PrettyPrintConfussionMatrix import PrettyPrintConfussionMatrix


def main ():
    
    # var parser
    parser = DefaultParser (description = 'Generate HAHA output')
    
    
    # @var model_resolver ModelResolver
    model_resolver = ModelResolver ()
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var tasks List
    tasks = ['task-1', 'task-2', 'task-3', 'task-4']
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    

    # @var results Dict Here we store the results for task for building the dataframe
    results = {}

    
    # For each task
    for task in tasks:
    
        def callback (feature_key, y_pred, model_metadata):
            """
            Callback for each task
            """
            results[task] = y_pred
    
    
        # @var dataset Dataset This is the custom dataset for evaluation purposes
        dataset = dataset_resolver.get (args.dataset, args.corpus, task, False)
        
        
        # Determine if we need to use the merged dataset or not
        dataset.filename = dataset.get_working_dir (task, 'dataset.csv')
            
        
        # @var df Ensure if we already had the data processed
        df = dataset.get ()
        
        
        # @var task_type String
        task_type = dataset.get_task_type ()
        
        
        # @var feature_resolver FeatureResolver
        feature_resolver = FeatureResolver (dataset)
        
        
        # Replace the dataset to contain only the test or val-set
        dataset.default_split = 'test'
        
        
        # @var model Model
        model = model_resolver.get ('deep-learning')
        model.set_dataset (dataset)
        model.is_merged (dataset.is_merged)
    
        
        # @var feature_combinations List
        # feature_combinations = [['lf', 'se', 'bf']]
        feature_combinations = [['lf', 'bf']] if task == 'task-1' else [['lf', 'se', 'bf']]
        
            
        # Indicate which features we are loading
        print ("loading features for task " + task + "...")
        
        
        # Load all the available features
        for features in feature_combinations:
        
            print (features)
            print ("-----------------")
        
            # Load features
            for feature_set in features:

                # @var feature_file String
                feature_file = feature_resolver.get_suggested_cache_file (feature_set, task_type)

            
                # @var features_cache String The file where the features are stored
                features_cache = dataset.get_working_dir (task, feature_file)

                
                # If the feautures are not found, get the default one
                if not Path (features_cache, cache_file = "").is_file ():
                    features_cache = dataset.get_working_dir (task, feature_set + '.csv')
                
                
                # Indicate what features are loaded
                print ("\t" + features_cache)
                if not Path (features_cache).is_file ():
                    print ("skip...")
                    continue
                    
                    
                # Set features
                model.set_features (feature_set, feature_resolver.get (feature_set, cache_file = features_cache))


        # Predict this feature set
        model.predict (
            using_official_test = True, 
            callback = callback, 
            use_train_val = task in ['task-1', 'task-2', 'task-3', 'task-4']
        )


        # Clear session
        model.clear_session ();


    # @var dataset Dataset Get the dataset for task 1 to retrieve the document IDs
    dataset = dataset_resolver.get (args.dataset, args.corpus, 'task-1', False)
    
    
    # Determine if we need to use the merged dataset or not
    dataset.filename = dataset.get_working_dir ('task-1', 'dataset.csv')
    
    
    # Replace the dataset to contain only the test or val-set
    dataset.default_split = 'test'
    
    
    # @var df_test DataFrame with test data the IDs 
    df_test = dataset.get ()
    
    
    # Fix regression to bound values
    results['task-2'] = np.where (results['task-2'] < 0, 0, results['task-2'])
    
    
    # @var result_df DataFrame
    result_df = pd.DataFrame (results)
    
    
    # Attach label. We merge the dataframes to ignore the index
    result_df = pd.concat ([pd.DataFrame (df_test['twitter_id']).reset_index (drop = True), result_df.reset_index (drop=True)], axis = 1, ignore_index = True)
    result_df.columns = ['id', 'is_humor', 'humor_rating', 'humor_mechanism', 'humor_target']
    

    # Adapt the dataframe
    # @see https://competitions.codalab.org/competitions/30090#learn_the_details-how-to-submit
    result_df = result_df.replace ({
        "is_humor": {"humor": 1, "non-humor": 0},
        "humor_mechanism": {"none": ""},
        "humor_target": {"none": ""}
    })
    
    # result_df.loc[result_df['is_humor'] == 0, 'humor_rating'] = .0
    

    result_df.to_csv (dataset.get_working_dir ('umuteam_run_6.csv'), index = False)

if __name__ == "__main__":
    main ()