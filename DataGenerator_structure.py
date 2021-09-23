class DataGenerator(keras_Sequence):                                            
                                                                                
    """                                                                         
    Parameters                                                                  
    ----------   
    list_IDs: list of strings
        List of strings (file names)
        
    labels: dict
        {"file name": "target"}
        
    onehot_encoder: scikit-learn OneHotEncoder
        Fitted with genres list in GTZAN dataset
        
    path_to_saved_datapoints: string
        Path where are stored preprocess datas created by GtzanPreprocessing
        
    datapoints_per_file: int
        Number of short audio extract in one save file
        
    dim: int or tuple
        Dimension of a datapoint 
        
    n_classes: int
        Number of classes (10 for GTZAN dataset)
        
    batch_size: int
        Number of files loaded per epoch step 
        
    n_channels: int
        Number of channels (for example RGB image, n_channels = 3)
        
    shuffle: bool
        To shuffle or not, the order of datapoints between epochs
        
    to_fit: bool
        To return or not, targets
        
    ML: bool
        To return targets as onehotvectors or labels
    
    only_targets: bool
        To return only targets
                                                                    
    Methods                                                                     
    -------                                                                     
    def __len__()                             
        Calculate the number of batchs  
        
    def __getitem__(index)
        Return batch data
    
    def on_epoch_end()
        Executed at initialisation and epoch end
    
    def __data_generation(list_IDs_batch)
        Call in __get_item__ to load as intended saved datas (unwrapped short extract contains in each files)                                                                                     
    """    
    
    def __init__(self, list_IDs, labels, onehot_encoder, path_to_saved_datapoints, datapoints_per_file, dim, n_classes, batch_size=32, n_channels=1, shuffle=False, to_fit=False, ML=False, only_targets=False):                                                    
        
        """
        self.parameters = parameters
        """ 
                                                                                                                                                                       
        self.on_epoch_end()                                                     
                                                                                
    def __len__(self):                                                                                                                                                
        length = int(np_floor(len(self.list_IDs) / self.batch_size))            
        return length
    
    def __getitem__(self, index):      
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]                                                                                             
        list_IDs_batch = [self.list_IDs[k] for k in indexes]                                                  
                                                                                                               
        if self.to_fit and self.only_targets == False:                          
            X, y = self.__data_generation(list_IDs_batch)                        
            return X, y                                                         
        elif self.only_targets:                                                 
            y = self.__data_generation(list_IDs_batch)                           
            return y                                                            
        else:                                                                   
            X = self.__data_generation(list_IDs_batch)                           
            return X                                                            
                                                                                
    def on_epoch_end(self):                                                                                        
        self.indexes = np_arange(len(self.list_IDs))                            
        if self.shuffle == True:                                                
            np_shuffle(self.indexes)                                                                                     
        
    def __data_generation(self, list_IDs_batch):
        """                                                                 
        Parameters                                                          
        -------      
        list_IDs_batch
            Path to folder where preprocessed datas will be saved
                                                                                
        Returns                                                             
        -------                                                             
           Generator
        """  
        
        """
        Declare signals (numpy empty array)
        Declare targets (numpy empty array) (default type for Deep Learning Approach, dtype="<U9" for Machine Learning Approach)
        For element in list_IDs_batch
            Load file -> signal , target
            Save in signals and targets array the loaded signal and target
        Return signals and targets  
        """
