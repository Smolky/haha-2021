import csv
import sys
import string
import numpy as np
import pandas as pd
import os
import config

from .Dataset import Dataset

class DatasetHaha (Dataset):
    """
    DatasetHaha
    
    Task description
    There are four subtasks, two of them are analogous to the subtasks in HAHA 2018 and 2019, 
    while the other two are new. Based on tweets written in Spanish, the following 
    subtasks are organized:

    (1) Humor Detection​: determining if a tweet is a joke or not (intended humor by the author or not). 
    The performance of this task will be measured using the F1 score of the ‘humorous’ class.
    
    (2) Funniness Score Prediction​: predicting a Funniness Score value for a tweet in a 5-star ranking, 
    assuming it is a joke. The performance of this task will be measured using the 
    root mean squared error of the funniness score.
    
    (3) Humor Mechanism Classification​: for a humorous tweet, predict the mechanism by which 
    the tweet conveys humor from a set of classes such as irony, wordplay or exaggeration. 
    In this task, only one class per tweet is allowed. The performance of this task will be 
    measured using the Macro-F1 score.
    
    (4) Humor Target Classification​: for a humorous tweet, predict the target 
    of the joke (what it is making fun of) from a set of classes such as racist jokes, 
    sexist jokes, etc. This task might be related to other tasks such as detection 
    of offensiveness or hate speech. In this case, there could be many classes associated 
    with a tweet, and also tweets that do not belong to any of the categories 
    (multi-label classification). The performance of this task will be measured 
    using the Macro-F1 score.

    @link https://competitions.codalab.org/competitions/30090#learn_the_details

    @extends Dataset
    """

    def __init__ (self, dataset, options, corpus = '', task = '', refresh = False):
        """
        @inherit
        """
        Dataset.__init__ (self, dataset, options, corpus, task, refresh)
        
    
    def compile (self):
        
        # @var dfs List list of DataFrames
        dfs = []
        
        
        # Load dataframes
        for index, dataframe in enumerate (['train.csv', 'val.csv', 'test.csv']):
        
            # @var df DataFrame
            df_split = pd.read_csv (self.get_working_dir ('dataset', dataframe))
            
            
            # @var split String
            split = 'train' if index == 0 else ('val' if index == 1 else 'test')
            
            
            # Determine split
            df_split = df_split.assign (__split = split)
            
        
            # Merge
            dfs.append (df_split)
        
        
        # Concat and assign
        df = pd.concat (dfs, ignore_index = True)
        
        
        # Remove blank lines
        df = df.replace (to_replace=[r"(.*)\.(\\t|\\n|\\r)", "(.*)\.(\t|\n|\r)"], value=[r"\1. ", r"\1. "], regex = True)
        df = df.replace (to_replace=[r"(\\t|\\n|\\r)", "(\t|\n|\r)"], value=[" ", " "], regex = True)
        
        
        # Reassign labels
        df = df.rename (columns = {
            "id": "twitter_id", 
            "text": "tweet",
            "is_humor": "label"
        })
        

        # Reassign labels
        df.loc[df['label'] == 1, 'label'] = 'humor'
        df.loc[df['label'] == 0, 'label'] = 'non-humor'
        
        
        # NOTE: Labels are not available on development split. For this, we will 
        # create a new split
        train, val = np.split (df.loc[df['__split'] == 'train'].sample (frac = 1), [
            int (.8 * len (dfs[0]))
        ])
        
        
        # Move val to another split
        df.loc[df['__split'] == 'val', '__split'] = 'official_val'
        
        
        # Reassign train and val
        df['__split'][train.index] = 'train'
        df['__split'][val.index] = 'val'
        
        
        # Store this data on disk
        self.save_on_disk (df)
        
        
        # Return
        return df
        
        
    def getDFFromTask (self, task, df):
        
        """
        Adjust the dataset for an specific task
        
        @param task string
        @param df DataFrame
        """
        
        # Do generic stuff, as label assignment.
        df = super ().getDFFromTask (task, df)
        
        
        # Adjust labels
        if 'task-3' == task:
            df['label'] = df['label'].fillna ('none')
            
            
            # For testing.
            # @todo. Remove none documents only for training
            # Get only a subset of none documents
            # df['label'] = df['label'].fillna ('none', limit = int (4800 / 12))
            
            
            # Remove the rest
            df = df.dropna (subset = ['label'])
            

        if 'task-4' == task:
            df = df.dropna (subset = ['label'])
            # df['label'] = df['label'].fillna ('none')
            
            # When this task is handled as classification, we keep the first tag
            if self.get_task_type () == 'classification':
                df['label'] = df['label'].str.split (';').str[0]
        
        
        return df