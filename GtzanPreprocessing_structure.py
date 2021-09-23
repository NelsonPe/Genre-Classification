class GtzanPreprocessing:                                                       
                                                                                
    """                                                                         
    Parameters                                                                  
    ----------                                                                  
    path_to_database : string                                                   
        path on hard drive to 'GTZAN/genres/' folder                            
                                                                                
    Methods                                                                     
    -------                                                                     
    def cut_one_signal(signal, extract_size, step)                              
        Return a signal cut in arrays of size <extract_size> every <step>    
        
    def resampling(signal, sampling_rate, resampling_rate)
        Return resample signal using librosa resampling function
    
    def feature_transform(signal, feature, rate=None)
        Return list of transformed short audio extract
    
    def pourc_to_ascii_loadingbar(pourc, bar_width=30)
        Generate a ascii loading bar [======>-----------------------]
    
    def follow_execution(func_name, start_time, last_print, i, signals, verbose=1)
        Display preprocessing progression
    
    def conversion_secondes_into_date(n_in_secondes)
        Convert a number of seconds into a string "%iy %im %iw %id %ih %im %is"
                                                                                
    def create_preloading_matrix(GTZAN_path, genres_list)                       
        Return a two columns preload_dataframe ['Paths', 'Targets']             
                                                                                
    def load_GTZAN_raw_signals(preload_dataframe)                               
        Return three lists: signals, sampling_rates, targets      
        
    def GTZAN_data_augmentation(signal, sampling_rate, target, window_time_width_in_seconds, time_step_in_seconds)
        Return data augmented signal
    
    def save_datapoint
        Save datapoint on hard drive using numpy save
                                                                                                                                                             
    def fit_transform(save_folder, window_time_width_in_seconds=5, time_step_in_seconds=0.5, new_sampling_rate=0, transformation="", path_to_resampled_dataset="")                                        
        Execute all preprocessing pipeline                                                              
                                                                                
    """                                                                         

    def __init__(self, path_to_database):                                                                             
                                                                                
    def cut_one_signal(self, signal, extract_size, step):     
        """
        Return list of signal[i*step:i*step+extract_size]
        """   
        
    def resampling(self, signal, sampling_rate, resampling_rate):                                                                                                         
        return librosa_resample(signal, sampling_rate, resampling_rate)        
    
    def feature_transform(self, signal, feature, rate=None):    
        """
        47 lines
        Apply librosa or numpy transformation to each short song extract contains in signal according to feature (a string, for example: mfcc, 2dmelspectrogram, ...)
        """
                                                                                
    def pourc_to_ascii_loadingbar(self, pourc, bar_width=30):  
        bar = list("[" + "."*bar_width + "]")                                   
        n = floor(pourc*bar_width)                                              
        for i in range(1,n+1):                                                  
            bar[i] = "="                                                        
        if n < bar_width and pourc != 0.0 :                                     
            bar[n+1] = ">"                                                                                                                        
        return "".join(bar)                                                     

    def follow_execution(self, func_name, start_time, last_print, i, signals, verbose=1):  
        """
        21 lines
        Display Execution information Remaining time, Progression
        using self.pourc_to_ascii_loadingbar() and self.conversion_secondes_into_date()
        """                                                                                                                                                                                 

    def conversion_secondes_into_date(self, n_in_secondes):      
        """
        Covert number of seconds (integer) into "%iy %im %iw %id %ih %im %is" (years months weeks days hours minutes seconds)
        """

    def create_preloading_matrix(self, song=""):                                      
        """
        Return DataFrame with columns ['Paths', 'genre'] for each song in GTZAN dataset
        """                                                                                                       

    def load_GTZAN_raw_signals(self, preload_datapoint_info):    
        """
        Return signal, sampling_rate and target for each song (row in preloading matrix)
        """
                                                 
    def GTZAN_data_augmentation(self, signal, sampling_rate, target, window_time_width_in_seconds, time_step_in_seconds):                                                                                                                                                    
        signals_cut = self.cut_one_signal(signal, int(sampling_rate*window_time_width_in_seconds), int(sampling_rate*time_step_in_seconds))                                                                         
        return signals_cut, [target]*len(signals_cut)                           

    def save_datapoint(self, datapoint_name, save_folder, signal, target, sampling_rate):                                                                                                                
        with open(save_folder + datapoint_name + ".npy", "wb") as f:             
            np_save(f, signal)                                                   
            np_save(f, target)                                                   
            np_save(f, sampling_rate)                                           

    def fit_transform(self, 
                      save_folder, 
                      window_time_width_in_seconds=5, 
                      time_step_in_seconds=0.5, 
                      new_sampling_rate=0,                               
                      transformation="",                                 
                      path_to_resampled_dataset=""):                                        
                                                                                
            """                                                                 
            Parameters                                                          
            -------      
            save_folder: string
                Path to folder where preprocessed datas will be saved
            
            window_time_width_in_seconds: int                                  
                Duration of rolling window used to make small extracts for data augmentation
                                                                                
            time_step_in_seconds: float                                        
                Step of rolling window    
                
            new_sampling_rate: int
                New sampling rate value, with new_sampling_rate == 0 no resampling is applied
                
            transformation: string
                Type of audio transformations, available types: ["2dmfcc", "mfcc", "2dmelspectrogram", "melspectrogram", "2dstft", "stft", "rfft"]                                     
                   
            path_to_resampled_dataset: string
                Path to presaved datas folder (skip loading from GTZAN dataset and resampling)
                                                                                
            Returns                                                             
            -------                                                             
               Nothing
            """                 
            
            """
            if path to (resampled or not) dataset not given :
                create preloading_dataframe (self.create_preloading_matrix)
                    load one song from preloading matrix 
                    apply resampling or not (self.resampling)
                    apply data augmentation (self.GTZAN_data_augmentation_one_song)
                    apply transformation or not (self.feature_transform_one_song)
                    save (self.save_datapoint)
                    display preprocessing progression (self.follow_execution)
                    append song target in labels dict
            else:
                load all presaved files path in path to dataset
                    load one file 
                    apply transformation or not (self.feature_transform_one_song)
                    save (self.save_datapoint)
                    display preprocessing progression (self.follow_execution)
                    append song target in labels dict
            Save labels dict     
            """
