"""
    To perform feature selection
    
    Right now this script only applies MinMaxScaler and a feature selection
    based on IG. These techniques were applied first for the LF; however, 
    we make another tests to another types of features. Note that we do not 
    apply this feature selection on the tokenizer (we)
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import VarianceThreshold

from tqdm import tqdm
from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser
from features.FeatureResolver import FeatureResolver


def main ():

    # var parser
    parser = DefaultParser (description = 'Feature selection')
    
    
    # Add parser
    parser.add_argument ('--features', 
        dest = 'features', 
        default = 'all', 
        help = 'Select the family or features to select', 
        choices = ['all', 'lf', 'be', 'se', 'ne', 'cf', 'bf', 'pr']
    )
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    dataset = dataset_resolver.get (args.dataset, args.corpus, args.task, False)
    
    
    # Determine if we need to use the merged dataset or not
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')


    # @var df Ensure if we already had the data processed
    df = dataset.get ()


    # @var df_train Ensure if we already had the data processed
    df_train = dataset.get_split (df, 'train')
    
    
    # @var task_type String
    task_type = dataset.get_task_type ()
    
    
    # @var indexes Dict the the indexes for each split
    indexes = {split: dataset.get_split (df, split).index for split in ['train', 'val', 'test']}
    
    
    # @var feature_resolver FeatureResolver
    feature_resolver = FeatureResolver (dataset)
    
    
    # @var available_features List
    available_features = ['lf', 'be', 'se', 'ne', 'cf', 'bf', 'pr'] if args.features == 'all' else [args.features]
    
    
    # @var scaler MinMaxScaler
    minmax_scaler = preprocessing.MinMaxScaler ()
    
    
    # @var robust_scaler MinMaxScaler
    robust_scaler = preprocessing.RobustScaler ()
    
    
    # @var variance_selector VarianceThreshold
    variance_selector = VarianceThreshold ()
    
    
    # Get every feature set
    for feature_set in (tqdm (available_features)):
    
        # @var features_cache String Retrieve the unprocessed features dataframe
        features_cache = dataset.get_working_dir (dataset.task, feature_set + '.csv')
        
        
        # Skip as those features does not exist
        if not Path (features_cache).is_file ():
            continue
        
        
        # @var transformer Retrieve the transformer for this feature set
        transformer = feature_resolver.get (feature_set, features_cache)
        
        
        # @var features_df DataFrame Retrieves all the features (they were previosly cached)
        features_df = transformer.transform ([]);
        
        
        # @var features_to_fit DataFrame Get the subset of the features in which we are 
        # going to fit our scalers
        features_to_fit = features_df[features_df.index.isin (indexes['train'])]
        
        
        # @var train_labels Series
        train_labels = df_train['label'].astype ('category').cat.codes
        
        
        # Fit the scalers on the train set
        minmax_scaler.fit (features_to_fit)
        robust_scaler.fit (features_to_fit)
        variance_selector.fit (features_to_fit)
        
        
        # @var minmax_fitted_features_df DataFrame All the features with MinMaxScaler
        minmax_fitted_features_df = pd.DataFrame (
            minmax_scaler.transform (features_df), 
            columns = features_df.columns, 
            index = features_df.index
        )
        
        
        # @var robust_fitted_features_df DataFrame All the features with RobustScaler
        robust_fitted_features_df = pd.DataFrame (
            robust_scaler.transform (features_df), 
            columns = features_df.columns, 
            index = features_df.index
        )

        
        # @var train_features_with_variance_columns List Remove duplicates without variance
        train_features_with_variance_columns = list (features_df.columns[variance_selector.get_support ()])
        
        
        # Feature selection for Classification task
        if 'classification' == task_type:
        
            # @var mutual_info_feature_selector SelectPercentile based on mutual information
            mutual_info_feature_selector = SelectPercentile (score_func = mutual_info_classif, percentile = 75)


            # @var anova_feature_selector SelectPercentile based on ANOVA F-value
            anova_feature_selector = SelectPercentile (score_func = f_classif, percentile = 75)

        
            # @var best_features_mutual_info_columns List
            best_features_mutual_info_columns = list (features_to_fit.columns[
                mutual_info_feature_selector
                    .fit (X = features_to_fit, y = train_labels)
                        .get_support ()
            ])
            
            
            # @var best_features_anova_columns List
            best_features_anova_columns = list (features_to_fit.columns[
                anova_feature_selector
                    .fit (X = features_to_fit, y = train_labels)
                        .get_support ()
            ])
            
            
            # Mix with the features with variance
            best_features_anova_columns = train_features_with_variance_columns + list (set (best_features_anova_columns) - set (train_features_with_variance_columns))
            best_features_mutual_info_columns = train_features_with_variance_columns + list (set (best_features_mutual_info_columns) - set (train_features_with_variance_columns))

            
            # Save features into disk
            # Save features into disk (information gain with minmax)
            minmax_fitted_features_df[best_features_mutual_info_columns].to_csv (
                dataset.get_working_dir (dataset.task, feature_set + '_minmax_ig.csv'), 
                index = False, 
                float_format = '%.5f'
            )
            
            
            # Save features into disk (ANOVA with minmax)
            minmax_fitted_features_df[best_features_anova_columns].to_csv (
                dataset.get_working_dir (dataset.task, feature_set + '_minmax_anova.csv'), 
                index = False, 
                float_format = '%.5f'
            )
            
            
            # Save features into disk (information gain with robust)
            robust_fitted_features_df[best_features_mutual_info_columns].to_csv (
                dataset.get_working_dir (dataset.task, feature_set + '_robust_ig.csv'), 
                index = False, 
                float_format = '%.5f'
            )
            
            
            # Save features into disk (ANOVA with robust)
            robust_fitted_features_df[best_features_anova_columns].to_csv (
                dataset.get_working_dir (dataset.task, feature_set + '_robust_anova.csv'), 
                index = False, 
                float_format = '%.5f'
            )
            
            
            # Save features into disk (information gain without any scaler)
            features_df[best_features_mutual_info_columns].to_csv (
                dataset.get_working_dir (dataset.task, feature_set + '_ig.csv'), 
                index = False, 
                float_format = '%.5f'
            )
            
            
            # Save features into disk (ANOVA without any scaler)
            features_df[best_features_anova_columns].to_csv (
                dataset.get_working_dir (dataset.task, feature_set + '_anova.csv'), 
                index = False, 
                float_format = '%.5f'
            )
            
            
        # Feature selection for Regression task
        if 'regression' == task_type:
            
            # @var regression_feature_selector SelectPercentile based on Univariate linear regression tests.
            regression_feature_selector = SelectPercentile (score_func = f_regression, percentile = 75)
            
            
            # @var best_features_univariate_regression_columns List
            best_features_univariate_regression_columns = list (features_to_fit.columns[
                regression_feature_selector
                    .fit (X = features_to_fit, y = train_labels)
                        .get_support ()
            ])
            
            
            # Mix with the features with variance
            best_features_univariate_regression_columns = train_features_with_variance_columns + list (set (best_features_univariate_regression_columns) - set (train_features_with_variance_columns))
            
            
            # Save
            minmax_fitted_features_df[best_features_univariate_regression_columns].to_csv (
                dataset.get_working_dir (dataset.task, feature_set + '_minmax_regression.csv'), 
                index = False, 
                float_format = '%.5f'
            )
            
            
            # Save features into disk (ANOVA with robust)
            robust_fitted_features_df[best_features_univariate_regression_columns].to_csv (
                dataset.get_working_dir (dataset.task, feature_set + '_robust_regression.csv'), 
                index = False, 
                float_format = '%.5f'
            )
            
            
            # Save features into disk (information gain without any scaler)
            features_df[best_features_univariate_regression_columns].to_csv (
                dataset.get_working_dir (dataset.task, feature_set + '_regression.csv'), 
                index = False, 
                float_format = '%.5f'
            )
            
            
        # MinMax Scaler
        minmax_fitted_features_df.to_csv (
            dataset.get_working_dir (dataset.task, feature_set + '_minmax.csv'), 
            index = False,
            float_format = '%.5f'
        )
        
        
        # Robust Scaler
        robust_fitted_features_df.to_csv (
            dataset.get_working_dir (dataset.task, feature_set + '_robust.csv'), 
            index = False,
            float_format = '%.5f'
        )

    

if __name__ == "__main__":
    main ()
