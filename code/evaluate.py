"""
    Evaluate a new text or a test dataset
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import argparse
import pandas as pd
import csv
import sklearn
import itertools

from pathlib import Path

from dlsdatasets.DatasetResolver import DatasetResolver
from dlsmodels.ModelResolver import ModelResolver
from features.FeatureResolver import FeatureResolver
from utils.Parser import DefaultParser
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error
from sklearn.metrics import ConfusionMatrixDisplay

from scipy.special import softmax
from utils.PrettyPrintConfussionMatrix import PrettyPrintConfussionMatrix

import seaborn as sn
import matplotlib.pyplot as plt



def get_feature_combinations (features):
    """
    List Get all the keys of the feature sets we are going to use
    Expand it to have features in isolation and combination (lf), (lf, se), ...
    
    @return List
    """
    
    # @var feature_combinations 
    feature_combinations = [key for key in features]
    feature_combinations = [list (subset) \
                                for L in range (1, len (feature_combinations) + 1) \
                                    for subset in itertools.combinations (feature_combinations, L)]
                                    

    return feature_combinations
    

def main ():

    # var parser
    parser = DefaultParser (description = 'Evaluate dataset')
    
    
    # @var model_resolver ModelResolver
    model_resolver = ModelResolver ()
    
    
    # @var confussion_matrix_pretty_printer PrettyPrintConfussionMatrix
    confussion_matrix_pretty_printer = PrettyPrintConfussionMatrix ()
    
    
    # Add model
    parser.add_argument ('--model', 
        dest = 'model', 
        default = model_resolver.get_default_choice (), 
        help = 'Select the family of algorithms to evaluate', 
        choices = model_resolver.get_choices ()
    )
    
    
    # @var choices List of list 
    choices = get_feature_combinations (['lf', 'se', 'be', 'we', 'ne', 'cf', 'bf', 'pr'])
    
    
    # Add features
    parser.add_argument ('--features', 
        dest = 'features', 
        default = 'all', 
        help = 'Select the family or features to evaluate', 
        choices = ['all'] + ['-'.join (choice) for choice in choices]
    )
    
    
    # Add features
    parser.add_argument ('--source', 
        dest = 'source', 
        default = 'test', 
        help = 'Determines the source to evaluate', 
        choices = ['all', 'train', 'test', 'val']
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
    
    
    # @var model Model
    model = model_resolver.get (args.model)
    model.set_dataset (dataset)
    model.is_merged (dataset.is_merged)
    
    
    # @var task_type String
    task_type = dataset.get_task_type ()


    # @var feature_resolver FeatureResolver
    feature_resolver = FeatureResolver (dataset)
    
    
    # Replace the dataset to contain only the test or val-set
    if args.source in ['train', 'val', 'test']:
        dataset.default_split = args.source

    
    
    # @var feature_combinations List
    feature_combinations = get_feature_combinations (model.get_available_features ()) if args.features == 'all' else [args.features.split ('-')]
    
    
    def callback (feature_key, y_pred, model_metadata):
        """
        
        """

        # @var y_real Remove NaN for regression tasks
        if task_type == 'regression':
            y_real = dataset.get ().dropna (subset = ['label'])['label']
            
        elif task_type == 'classification':
            y_real = dataset.get ()['label']
            
        elif task_type == 'multi_label':
            y_real = dataset.get ()['label'].str.split ('; ')
            
            
            # Create a binarizer
            # @todo Save binarizer when training
            lb = sklearn.preprocessing.MultiLabelBinarizer ()
            
            
            # Fit the multi-label binarizer
            y_real = lb.fit_transform (y_real)
            
        
        # @var labels List
        labels = dataset.get_available_labels ()
        
        
        # @var y_real_labels_available boolean
        y_real_labels_available = not pd.isnull (y_real).all ()
        if 'regression' == task_type:
            y_real_labels_available = None
        
        
        # @var report DataFrame|None
        report = pd.DataFrame (classification_report (
            y_true = y_real, 
            y_pred = y_pred, 
            digits = 5,
            output_dict = True
        )).T if y_real_labels_available else None
        
        
        
        # @var probabilities_df None Assign probabilities
        probabilities_df = None
        if 'probabilities' in model_metadata and args.model != 'ensemble' and 'classification' == task_type:
            
            # Attach the labels
            probabilities_df = pd.DataFrame (model_metadata['probabilities'], columns = labels if len (labels) > 2 else [labels[0]])
            
            
            # For case of binary labels include the other one
            if len (labels) <= 2:
                probabilities_df[labels[1]] = 1 - model_metadata['probabilities']
            
            
            # Include the real label
            probabilities_df = probabilities_df.assign (label = y_real.reset_index (drop = True))
            
            
            # Include information regarding the feature set employed
            probabilities_df = probabilities_df.assign (features = feature_key)
            
            
            # Reorder
            probabilities_df = probabilities_df[['features'] + [col for col in probabilities_df.columns if col != 'features']]
        
        
        # @var cm confusion matrix|none
        cm = None
        
        if y_real_labels_available:
            if 'classification' == task_type:
                cm = sklearn.metrics.confusion_matrix (
                    y_true = y_real, 
                    y_pred = y_pred, 
                    labels = labels, 
                    normalize = 'true'
                )
            
            elif 'multi_label' == task_type:
                cm = sklearn.metrics.multilabel_confusion_matrix (
                    y_true = y_real, 
                    y_pred = y_pred
                )
            
            
            # Create confusion matrix as image
            plt.clf ()
            ax = plt.subplot ()
            sn.set (font_scale = .5)
            heatmap = sn.heatmap (cm * 100, annot = True, fmt = '.0f', annot_kws = {'size': 'small'}, cbar = None)
            ax.xaxis.set_ticklabels (labels, rotation = 90); 
            ax.yaxis.set_ticklabels (labels, rotation = 30);
            for t in ax.texts: t.set_text (t.get_text() + "%")
            

        
        # @var tables_path String
        tables_path = dataset.get_working_dir (dataset.task, 'results', args.source, args.model, feature_key, 'classification_report.html')
        
        
        # @var report_path String
        report_path = dataset.get_working_dir (dataset.task, 'results', args.source, args.model, feature_key, 'classification_report.latex')
        
        
        # @var confusion_matrix_path String
        confusion_matrix_path = dataset.get_working_dir (dataset.task, 'results', args.source, args.model, feature_key, 'confusion_matrix.latex')


        # @var confusion_matrix_heatmap_path String
        confusion_matrix_heatmap_path = dataset.get_working_dir (dataset.task, 'results', args.source, args.model, feature_key, 'confusion_matrix.pdf')
        
        
        # @var probabilities_path String
        probabilities_path = dataset.get_working_dir (dataset.task, 'results', args.source, args.model, feature_key, 'probabilities.csv')
        
        
        # @var weights_path String
        weights_path = dataset.get_working_dir (dataset.task, 'results', args.source, args.model, feature_key, 'weights.csv')


        # @var predictions_path String
        predictions_path = dataset.get_working_dir (dataset.task, 'results', args.source, args.model, feature_key, 'predictions.csv')
        
        
        print ()
        print (feature_key)
        print ("================")
        
        print ("created_at")
        print ("-----------------")
        print (model_metadata['created_at'])
        
        
        # Report
        if report is not None:
            print ()
            print ("classification report")
            print ("-----------------")
            print (report.to_markdown ())
        
            # Store reports
            report.to_latex (report_path, index = True)
            report.to_html (tables_path)

        
        # Confusion matrix
        if cm is not None and task_type == 'classification':
            print ()
            print ("confusion matrix")
            print ("-----------------")
            confussion_matrix_pretty_printer.print (cm, labels)
            pd.DataFrame (cm).to_latex (confusion_matrix_path, index = True)
        
            plt.savefig (confusion_matrix_heatmap_path, bbox_inches = 'tight')

        
        # Save probabilities
        if probabilities_df is not None:
            probabilities_df.to_csv (probabilities_path, index = False)
        
        
        print ()
        print ('probabilities')
        print ("-----------------")
        print (model_metadata['probabilities'])


        if y_real_labels_available:
            print ()
            print ('real')
            print ("-----------------")
            print (y_real)
        
        
            print ()
            print ('precision_recall_fscore_support')
            print ("-----------------")

            print (precision_recall_fscore_support (
                y_true = y_real, 
                y_pred = y_pred, 
                average = 'micro', 
                labels = labels if task_type == 'classification' else None)
            )

        if 'weights' in model_metadata:
            pd.DataFrame (model_metadata['weights'], index=[0]).to_csv (weights_path, index = False)
        
        
        # @var df_split Ensure if we already had the data processed
        df_split = dataset.get ()
    
        
        # Save predictions
        pd.DataFrame ({
            'id': df_split['twitter_id'],
            'tweet': df_split['tweet_clean'],
            'y_pred': y_pred,
            'y_real': y_real
        }).to_csv (predictions_path, index = False)
        
    
        if 'regression' == task_type: 
            print ('rmse ' + str (mean_squared_error (y_true = y_real, y_pred = y_pred)))
        

    # Load all the available features
    for features in feature_combinations:
        
        # Indicate which features we are loading
        print ("loading features...")
        
        for feature_set in features:

            # @var feature_file String
            feature_file = feature_resolver.get_suggested_cache_file (feature_set, task_type)

        
            # @var features_cache String The file where the features are stored
            features_cache = dataset.get_working_dir (args.task, feature_file)

            
            # If the feautures are not found, get the default one
            if not Path (features_cache, cache_file = "").is_file ():
                features_cache = dataset.get_working_dir (args.task, feature_set + '.csv')
            
            
            # Indicate what features are loaded
            print ("\t" + features_cache)
            if not Path (features_cache).is_file ():
                print ("skip...")
                continue
            
            
            # Set features
            model.set_features (feature_set, feature_resolver.get (feature_set, cache_file = features_cache))
    
    
    # Predict this feature set
    model.predict (using_official_test = True, callback = callback)
    
    
    # Clear session
    model.clear_session ();
    



if __name__ == "__main__":
    main ()
