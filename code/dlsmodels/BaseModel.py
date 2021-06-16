import itertools


class BaseModel ():
    """
    BaseModel
    
    @todo Separate models and merge at the end to have better sampling of the classifiers
    @todo Create ensemble model from different classifiers
    @todo Allow to make predictions easly for the winning models
    """
    
    # @var all_available_features List
    all_available_features = []
    
    
    # @var features Here we store the transformers we will use in our work
    features = {}
    
    
    # @var is_merged boolean
    is_merged = False
    
    
    def has_external_features (self):
        """
        @inherit
        """        
        return False
    
    
    def is_merged (self, is_merged):
        self.is_merged = is_merged
        
    
    def get_available_features (self):
        """
        get_available_features
        
        @return List
        """
        return self.all_available_features
    
    
    def set_dataset (self, dataset):
        """
        set_dataset
        
        @param dataset Dataset
        """
        self.dataset = dataset
    
    
    def set_features (self, key, features):
        """
        set_features
        
        @param key string
        @features Transformer
        """
        self.features[key] = features;
        
        
    def clear_session (self):
        """
        clear_session
        """
        self.features = {};
        self.all_available_features = []
        self.is_merged = False
        
    
    def get_feature_combinations (self):
        """
        List Get all the keys of the feature sets we are going to use
        Expand it to have features in isolation and combination (lf), (lf, se), ...
        
        @return List
        """
        
        # @var force_order_indices Map
        force_order_indices = {c: i for i, c in enumerate ("lf se we be ne bf cf fe".split())}
        
        
        # @var keys List
        keys = self.features.keys ()
        keys = sorted (keys, key = force_order_indices.get)
        
        return tuple (keys)

    @property
    def train (self):
        raise NotImplementedError ("Subclasses should implement this!")    
        
        
    @property
    def predict (self):
        raise NotImplementedError ("Subclasses should implement this!")    