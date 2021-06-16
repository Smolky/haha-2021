"""
DatasetResolver

@author José Antonio García-Díaz <joseantonio.garcia8@um.es>
@author Rafael Valencia-Garcia <valencia@um.es>
"""

import json
import sys
import os.path


from .Dataset import Dataset
from .DatasetPoliCorpus import DatasetPoliCorpus
from .DatasetHahackathon import DatasetHahackathon
from .DatasetTASS import DatasetTASS
from .DatasetPANAuthorProfiling2018 import DatasetPANAuthorProfiling2018
from .DatasetPANAuthorProfiling2019 import DatasetPANAuthorProfiling2019
from .DatasetPANAuthorProfiling2021 import DatasetPANAuthorProfiling2021
from .DatasetMisoCorpus import DatasetMisoCorpus
from .DatasetCH import DatasetCH
from .DatasetSatire import DatasetSatire
from .DatasetCheckThat import DatasetCheckThat
from .DatasetMeOffendES import DatasetMeOffendES
from .DatasetMeOffendMX import DatasetMeOffendMX
from .DatasetExist import DatasetExist
from .DatasetEMOEval import DatasetEMOEval
from .DatasetHatEval import DatasetHatEval
from .DatasetAMI import DatasetAMI
from .DatasetHaha import DatasetHaha
from .DatasetHaterNet import DatasetHaterNet
from .DatasetMEXA3T import DatasetMEXA3T


class DatasetResolver ():
    """
    DatasetResolver
    """
    
    def get (self, dataset, corpus = '', task = '', refresh = False):
        """
        @param dataset string
        @param corpus string|null
        @param refresh string|null
        @param task boolean
        """
    
        # Load configuration of the dataset
        with open ('../config/dataset.json') as json_file:
            
            # @var dataset_options Dict
            dataset_options = json.load (json_file)[dataset]
            
            
            # If corpus is not supplied, then get the first element of our dataset
            corpus = list (dataset_options.keys())[0] if not corpus else corpus
            
            
            # @var corpus_options Retrieve data from configuration
            corpus_options = dataset_options[corpus]
        
        
        # @var args Dict 
        args = {
            'dataset': dataset, 
            'options': corpus_options, 
            'corpus': corpus, 
            'task': task, 
            'refresh': refresh
        }
        
        
        # Default implementation
        if not 'datasetClass' in corpus_options:
            return Dataset (**args)


        class_name = globals()[corpus_options['datasetClass']]
        return class_name (**args)

    