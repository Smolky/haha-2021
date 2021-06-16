"""
    Show dataset correlation
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys
import config
import bootstrap
import os.path
import pandas as pd
import numpy as np
import scipy.stats as ss
import seaborn as sn
import matplotlib.pyplot as plt

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser
from scipy.stats import chisquare


def cramers_v (confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
        
        @link https://stackoverflow.com/questions/46498455/categorical-features-correlation/46498792#46498792
        @link https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))    



def main ():
    
    # var parser DefaultParser
    parser = DefaultParser (description = 'Compile dataset', defaults = {
        'dataset': 'haha',
        'corpus': '2021-es'
    })
    

    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var resolver Resolver
    resolver = DatasetResolver ()
    
    
    # @var dataset_task_3 Dataset
    dataset_task_3 = resolver.get (args.dataset, args.corpus, 'task-3', False)
    dataset_task_3.filename = dataset_task_3.get_working_dir (args.task, 'dataset.csv')

    # @var dataset_task_4 Dataset
    dataset_task_4 = resolver.get (args.dataset, args.corpus, 'task-4', False)
    dataset_task_4.filename = dataset_task_4.get_working_dir (args.task, 'dataset.csv')
    
    
    # @var df Dataframe
    df_task_3 = dataset_task_3.get ()    
    df_task_4 = dataset_task_4.get ()    
    
    
    # @var df_task_3_labels
    df_task_3_labels = df_task_3.loc[df_task_3['label'] != 'none']['label']
    
    
    # @var df_task_4_labels
    df_task_4_labels = df_task_4.loc[df_task_4['label'] != 'none']
    df_task_4_labels = df_task_4['label'].str.split (';').str[0]
    df_task_4_labels = df_task_4_labels.reindex (df_task_3_labels.index)
    print (df_task_4_labels)
    
    
    # Create dataframe
    corref_df = pd.concat ([df_task_3_labels, df_task_4_labels], axis = 1).fillna ('none')
    corref_df.columns = ['mechanism', 'target']
    
    
    # @var confusion_matrix DataFrame
    confusion_matrix = pd.crosstab (
        index = corref_df['mechanism'], 
        columns = corref_df['target'],
        normalize = False
    )
    
    
    # Remove none column
    confusion_matrix = confusion_matrix.drop (['none'], axis = 1)
    
    
    targets = sorted (corref_df['target'].unique ())
    mechanisms = sorted (corref_df['mechanism'].unique ())
    targets = [target for target in targets if target != 'none'] 
    mechanisms = [mechanism for mechanism in mechanisms if mechanism != 'none'] 
    
    
    # Create confusion matrix as image
    plt.clf ()
    ax = plt.subplot ()
    sn.set (font_scale = .5)
    heatmap = sn.heatmap (confusion_matrix, annot = True, fmt = '.0f', annot_kws = {'size': 'small'}, cbar = None)
    ax.xaxis.set_ticklabels (targets, rotation = 90); 
    ax.yaxis.set_ticklabels (mechanisms, rotation = 30);
    ax.set_ylabel ('')
    ax.set_xlabel ('')
    

    print (targets)
    print (mechanisms)
    plt.savefig ('haha-correlation-mechanism-target.pdf', bbox_inches = 'tight')
    

if __name__ == "__main__":
    main ()