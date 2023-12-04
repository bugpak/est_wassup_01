import pandas as pd
import numpy as np
from imblearn.under_sampling import TomekLinks

class Imbalance_Module:
    def __init__(self):
        self.ada = TomekLinks(sampling_strategy='majority')
        
    def resample(self,df):
        print('resampling start!')
        y = df['ECLO']
        X_res, y_res = self.ada.fit_resample(df, y.squeeze())
        X_res, y_res = self.ada.fit_resample(X_res, y_res.squeeze())
        X_res, y_res = self.ada.fit_resample(X_res, y_res.squeeze())
        X_res, y_res = self.ada.fit_resample(X_res, y_res.squeeze())
        X_res, y_res = self.ada.fit_resample(X_res, y_res.squeeze())
        
        return X_res
       