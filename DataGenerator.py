from numpy import floor as np_floor 
from numpy import arange as np_arange                                           
from numpy.random import shuffle as np_shuffle   
from numpy import empty as np_empty   
from keras.utils import Sequence as keras_Sequence 
from Signals_approaches.Preprocess import load_np_array_with_sampling_rate

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
    def __init__(self,                                                          
                 list_IDs,                                                      
                 labels,                                                        
                 onehot_encoder,                                                
                 path_to_saved_datapoints,                                      
                 datapoints_per_file,                                           
                 batch_size=32,                                                 
                 dim=(32,32,32),                                                
                 n_channels=1,                                                  
                 n_classes=10,                                                  
                 shuffle=False,                                                 
                 to_fit=False,                                                  
                 ML=False,                                                      
                 only_targets=False):                                                    
                                                                                                                                       
        self.dim = dim                                                          
        self.datapoints_per_file = datapoints_per_file                          
        self.batch_size = batch_size                                            
        self.labels = labels                                                    
        self.onehot_encoder = onehot_encoder                                    
        self.path_to_saved_datapoints = path_to_saved_datapoints                
        self.list_IDs = list_IDs                                                
        self.n_channels = n_channels                                            
        self.n_classes = n_classes                                              
        self.shuffle = shuffle                                                  
        self.to_fit = to_fit                                                    
        self.ML = ML                                                            
        self.verbose = verbose                                                  
        self.only_targets = only_targets                                                                          
        self.indexes = np_arange(len(self.list_IDs))                                                                                                                        
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
           
        Description
        -----------
        Declare signals (numpy empty array)
        Declare targets (numpy empty array) (default type for Deep Learning Approach, dtype="<U9" for Machine Learning Approach)
        For element in list_IDs_batch
            Load file -> signal , target
            Save in signals and targets array the loaded signal and target
        Return signals and targets  
        """                                                        
        if self.only_targets and self.to_fit:                                   
            raise Exception("You cannot make to_fit and only_targets True at the same time (asking outputs x,y and y at the same time)")
        else:                                                                   
            pass                                                                
                                                                                
        if self.only_targets == False:                                          
            if type(self.dim) == type(1):                                       
                X = np_empty((self.batch_size*self.datapoints_per_file, self.dim, self.n_channels))
            else:                                                               
                X = np_empty((self.batch_size*self.datapoints_per_file, *self.dim, self.n_channels))
        else:                                                                   
            pass                                                                
                                                                                
        if self.to_fit or self.only_targets:                                    
            if self.ML:                                                         
                y = np_empty((self.batch_size*self.datapoints_per_file, self.n_classes), dtype='<U9')
            else:                                                               
                y = np_empty((self.batch_size*self.datapoints_per_file, self.n_classes))
        else:                                                                   
            pass                                                                
        # Generate data                                                         
        for i, ID in enumerate(list_IDs_temp):                                  
            # Store sample                                                      
            inputs_in_file, targets, sr = load_np_array_with_sampling_rate(self.path_to_saved_datapoints + ID)
            number_of_strings_in_file = 0                                       
            for j, input_in_file in enumerate(inputs_in_file):                  
                shape = input_in_file.shape                                     
                if self.only_targets == False:                                  
                    X[self.datapoints_per_file*i+j,] = input_in_file.reshape((*shape, self.n_channels))
                else:                                                           
                    pass                                                        
                if self.to_fit or self.only_targets:                            
                    if self.ML == True:                                         
                        y[self.datapoints_per_file*i+j,] = targets[j]           
                    else:                                                       
                        y[self.datapoints_per_file*i+j,] = self.onehot_encoder.transform([[targets[j]]]).toarray()[0]
                else:                                                           
                    pass
                number_of_strings_in_file +=1                                   
            assert number_of_strings_in_file == self.datapoints_per_file        
        if self.to_fit and self.only_targets == False:                          
            return X, y                                                         
        elif self.only_targets:                                                 
            return y                                                            
        else:                                                                   
            return X                     
