import pandas as pd
import numpy as np
from imblearn.under_sampling import TomekLinks, OneSidedSelection, RandomUnderSampler
from imblearn.combine import SMOTETomek

class Imbalance_Module:
    '''
    function for resampling
    '''
    def __init__(self):
        self.tmk = TomekLinks(sampling_strategy='majority')
        self.rus = RandomUnderSampler(sampling_strategy='majority', random_state=2023)
        self.oss = OneSidedSelection(sampling_strategy='majority', random_state=2023)
        self.stmk = SMOTETomek(sampling_strategy='auto')
        
    def resample(self,df):
        print('resampling start!')
        y = df['ECLO']
        for i in range(3):
            df, y = self.oss.fit_resample(df, y)
        
        return df
       