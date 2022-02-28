from timeit import default_timer
from os import listdir
from pandas import DataFrame, concat
from numpy import array as np_array
from numpy import max as np_max
from numpy import save as np_save
from math import floor
from IPython.display import clear_output
from librosa import load as librosa_load
from librosa import resample as librosa_resample                                   
from librosa import stft as librosa_stft                                                                        
from librosa import power_to_db as librosa_power_to_db                          
from librosa.feature import mfcc as librosa_mfcc                                
from librosa.feature import melspectrogram as librosa_melspectrogram   

from Signals_approaches.Preprocess import load_np_array_with_sampling_rate

class GtzanPreprocessing:                                                                                                                        

    def __init__(self,                                                          
                 path_to_database):                                             
                                                                                
        self.path_to_database = path_to_database                                                                            
                                                                                
    def cut_one_signal(self,                                                    
                   signal,                                                      
                   extract_size,                                                
                   step):                                                       
                                                                                
        signals = []                                                            
        i=0                                                                     
        while i*step+extract_size <= len(signal):                               
            starting_point = i*step                                             
            signals.append(signal[starting_point:starting_point+extract_size])  
            i = i + 1                                                           
                                                                                
        return signals  
        
    def resampling_one_song(self,                                               
                            signal,                                             
                            sampling_rate,                                      
                            resampling_rate):                                   
                                                                                
        return librosa_resample(signal, sampling_rate, resampling_rate)     
    
    def feature_transform_one_song(self,                                        
                                   signal,                                      
                                   feature,                                     
                                   rate=None):                                  
                                                                                
        featured_one_song_signals = []                                          
        for signal_unwrapped in signal:                                         
            signal_unwrapped = signal_unwrapped.reshape([-1])                   
            if feature == "2dmfcc":                                             
                assert rate is not None, "2d mfcc sampling rate is not defined" 
                featured_signal = librosa_mfcc(y=signal_unwrapped, sr=rate, n_mfcc=128, hop_length=128)
            elif feature == "mfcc":                                             
                assert rate is not None, "mfcc sampling rate is not defined"    
                featured_signal = librosa_mfcc(y=signal_unwrapped, sr=rate, n_mfcc=128, hop_length=128).ravel()
            elif feature == "2dmelspectrogram":                                 
                assert rate is not None, "2d melspectrogram sampling rate is not defined"
                featured_signal = librosa_power_to_db(librosa_melspectrogram(y=signal_unwrapped, sr=rate, n_mels = 509), ref=np_max)
            elif feature == "melspectrogram":                                   
                assert rate is not None, "melspectrogram sampling rate is not defined"
                featured_signal = librosa_power_to_db(librosa_melspectrogram(y=signal_unwrapped, sr=rate, n_mels = 509), ref=np_max).ravel()
            elif feature == "2dstft":                                           
                featured_signal = librosa_power_to_db(np_abs(librosa_stft(y=signal_unwrapped, hop_length = 1024)), ref=np_max)
            elif feature == "stft":                                             
                featured_signal = librosa_power_to_db(np_abs(librosa_stft(y=signal_unwrapped, hop_length = 1024)), ref=np_max).ravel()
            elif feature == "rfft":                                             
                featured_signal = librosa_power_to_db(np_abs(np_rfft(signal_unwrapped)), ref=np_max)
            else:                                                               
                raise ValueError(f"Unknown feature {feature}")                  
            featured_one_song_signals.append(featured_signal)                   
                                                                                
        return featured_one_song_signals
                                                                                
    def pourc_to_ascii_loadingbar(self,                                         
                                  pourc,                                        
                                  bar_width=30):                                
                                                                                
        bar = list("[" + "."*bar_width + "]")                                   
        n = floor(pourc*bar_width)                                              
        for i in range(1,n+1):                                                  
            bar[i] = "="                                                        
        if n < bar_width and pourc != 0.0 :                                     
            bar[n+1] = ">"                                                      
                                                                                
        return "".join(bar)                                                     

    def follow_execution(self,                                                  
                         func_name,                                             
                         start_time,                                            
                         last_print,                                            
                         i,                                                     
                         signals,                                               
                         verbose=1):                                            
                                                                                
        #Display resampling progression and remaining time                      
        current_time = default_timer()                                          
        if (current_time - last_print) > 0.05:                                  
            remaining_time = ((current_time - start_time)/(i+1)) * (len(signals) - (i+1)) #remaining_time = average_one_iteration_time * remaining_iterations
            message = "Progression: %.2f" % (100*(i+1)/len(signals))            
            if verbose == 1:                                                    
                clear_output(wait=True)                                         
                display('Executing ' + func_name + '. ' + self.pourc_to_ascii_loadingbar((i+1)/len(signals)) + " " + message + "% Remaining time: " + self.conversion_secondes_into_date(remaining_time))
            else:                                                               
                pass                                                            
            return current_time                                                 
        else:                                                                   
            return last_print                                                                                                                                                                             

    def conversion_secondes_into_date(self,                                     
                                      n_in_secondes):                           
                                                                                
        secondes_in_minute, secondes_in_hour, secondes_in_day, secondes_in_week, secondes_in_month, secondes_in_year = 60, 60*60, 60*60*24, round(60*60*24*7), round(60*60*24*(365.25/12)), round(60*60*24*365.25)
                                                                                
        m_secondes = "%is" % (n_in_secondes%60)                                 
                                                                                
        if n_in_secondes > (secondes_in_minute-1):                              
            m_minutes = "%im " % ((n_in_secondes % secondes_in_hour) // secondes_in_minute)
        else:                                                                   
            m_minutes = ""                                                      
                                                                                
        if n_in_secondes > (secondes_in_hour-1):                                
            m_hours = "%ih " % ((n_in_secondes % secondes_in_day) // secondes_in_hour)
        else:                                                                   
            m_hours = ""                                                        
                                                                                
        if n_in_secondes > (secondes_in_day-1):                                 
            m_days = "%id " % ((n_in_secondes % secondes_in_week) // secondes_in_day)
        else:                                                                   
            m_days = ""                                                         
                                                                                
        if n_in_secondes > (secondes_in_week-1):                                
            m_weeks = "%iw " % ((n_in_secondes % secondes_in_month) // secondes_in_week)
        else:                                                                   
            m_weeks = ""                                                        
                                                                                
        if n_in_secondes > (secondes_in_month-1):                               
            m_months = "%im " % ((n_in_secondes % secondes_in_year) // secondes_in_month)
        else:                                                                   
            m_months = ""                                                       
                                                                                
        if n_in_secondes > (secondes_in_year-1):                                
            m_years = "%iy " % (n_in_secondes // secondes_in_year)              
        else:                                                                   
            m_years = ""                                                        
                                                                                
        return m_years + m_months + m_weeks + m_days + m_hours + m_minutes + m_secondes
                 
    def create_preloading_matrix(self,                                          
                                 song=""):                                                                                                            
        path_genre_matrix = DataFrame()                                         
        genres_list = [x for x in listdir(self.path_to_database) if x[-3:]!='.mf']
        for genre in genres_list:                                               
            folder_path = self.path_to_database+genre+'/'                       
            wav_files = [x for x in listdir(folder_path) if x[-4:]=='.wav']     
            for file_name in wav_files:                                         
                if song != "":                                                  
                    if file_name == song:                                       
                        d = {'Paths': [folder_path+file_name], 'genre': [genre]}
                        new_line = DataFrame(d)                                 
                        path_genre_matrix = concat([path_genre_matrix, new_line])
                    else:                                                       
                        pass                                                    
                else:                                                           
                    d = {'Paths': [folder_path+file_name], 'genre': [genre]}    
                    new_line = DataFrame(d)                                     
                    path_genre_matrix = concat([path_genre_matrix, new_line])   
            path_genre_matrix = path_genre_matrix.reset_index().drop(['index'],axis=1)
                                                                                
        return path_genre_matrix   

    def load_GTZAN_raw_signals_one_song(self,                                   
                                        preload_datapoint_info):                
        expected_signal_length = 30 * 22050                                     
        target = preload_datapoint_info[1]                                      
        signal, sampling_rate = librosa_load(preload_datapoint_info[0])         
                                                                                
        signals_resized = []                                                    
                                                                                
        if len(signal) >= expected_signal_length:                               
            signal_resized = signal[:expected_signal_length]                    
        else:                                                                   
            signal_resized = np_array(list(signal)+ [0.0]*(expected_signal_length-len(signal)))
                                                                                
        return signal_resized, sampling_rate, target
                                                 
    def GTZAN_data_augmentation_one_song(self,                                  
                                         signal,                                
                                         sampling_rate,                         
                                         target,                                
                                         window_time_width_in_seconds,          
                                         time_step_in_seconds):                 
                                                                                
                                                                                
        signals_cut = self.cut_one_signal(signal,                               
                                          int(sampling_rate*window_time_width_in_seconds),
                                          int(sampling_rate*time_step_in_seconds))
                                                                                
        return signals_cut, [target]*len(signals_cut)                             

    def save_datapoint(self,                                                    
                       datapoint_name,                                          
                       save_folder,                                             
                       signal,                                                  
                       target,                                                  
                       sampling_rate):                                          
                                                                                
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
                      path_to_resampled_dataset="",                      
                      one_song="",                                       
                      verbose=1):                                        
                                                                                                                                  
            labels_dict_keys, labels_dict_value = [], []                        
            if path_to_resampled_dataset == "":                                 
                if one_song != "":                                              
                    dataframe_pathsongs_targets = self.create_preloading_matrix(song=one_song)
                else:                                                           
                    dataframe_pathsongs_targets = self.create_preloading_matrix()
                start_time, last_print = default_timer(), 0.0                   
                for i, datapoint_index in enumerate(dataframe_pathsongs_targets.index):
                    signal, sampling_rate, target = self.load_GTZAN_raw_signals_one_song(dataframe_pathsongs_targets.iloc[datapoint_index])
                                                                                
                    if new_sampling_rate != 0:                                  
                        final_signal = self.resampling_one_song(signal, sampling_rate, new_sampling_rate)
                        del signal                                              
                        final_sampling = new_sampling_rate                      
                    else:                                                       
                        final_signal = signal
                    final_signal, final_target = self.GTZAN_data_augmentation_one_song(final_signal,
                                                                                       final_sampling,
                                                                                       target,
                                                                                       window_time_width_in_seconds,
                                                                                       time_step_in_seconds)
                    del target                                                  
                                                                                
                    if transformation != "":                                    
                        final_signal = self.feature_transform_one_song(final_signal,
                                                                       transformation,
                                                                       rate = final_sampling)
                    else:                                                       
                        pass                                                    
                    datapoint_id = "id-" + str(datapoint_index)                 
                    self.save_datapoint(datapoint_id, save_folder, final_signal, final_target, final_sampling)
                    last_print = self.follow_execution("Preprocessing without load",
                                                       start_time,              
                                                       last_print,              
                                                       i,                       
                                                       dataframe_pathsongs_targets.index,
                                                       verbose=verbose)         
                    labels_dict_keys.append(datapoint_id)                       
                    labels_dict_value.append(final_target)                      
                                                                                
            else:                                                               
                saved_datapoint_files = [x for x in listdir(path_to_resampled_dataset) if x[:3]=="id-"]
                start_time, last_print = default_timer(), 0.0                   
                for i, saved_datapoint_file in enumerate(saved_datapoint_files):
                    try:                                                        
                        final_signal, final_target, final_sampling = load_np_array_with_sampling_rate(
                            path_to_resampled_dataset + saved_datapoint_file.split('.')[0]
                        )                                                       
                    except:                                                     
                        raise Exception("Cannot load this datapoint: " + saved_datapoint_file + " from " + path_to_resampled_dataset)
                                                                                
                    if transformation != "":                                    
                        final_signal = self.feature_transform_one_song(final_signal,
                                                                       transformation,
                                                                       rate = final_sampling)
                    else:                                                       
                        pass                                                    
                                                                                
                    self.save_datapoint(saved_datapoint_file.split('.')[0], save_folder, final_signal, final_target, final_sampling)
                    last_print = self.follow_execution("Preprocessing with load",
                                                       start_time,              
                                                       last_print,              
                                                       i,                       
                                                       saved_datapoint_files,   
                                                       verbose=verbose)         
                    labels_dict_keys.append(saved_datapoint_file.split('.')[0]) 
                    labels_dict_value.append(final_target)                      
            with open(save_folder+"labels.npy", 'wb') as f:                     
                np_save(f, labels_dict_keys)                                    
                np_save(f, labels_dict_value)  
