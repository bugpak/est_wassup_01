import pandas as pd
import numpy as np
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from copy import deepcopy
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler

class Encoder_Module:
    def __init__(self):
        self.label = LabelEncoder()
        self.onehot = OneHotEncoder()
        
    def encoder(self, df:pd.DataFrame, enc:str='label'):
        if enc == 'label':
            df.loc[:,:] = df.loc[:,:].apply(self.label.fit_transform)
        if enc == 'onehot':
            df.loc[:,:] = df.loc[:,:].apply(self.onehot.fit_transform)
        
        return df