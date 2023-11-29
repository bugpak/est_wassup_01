import pandas as pd
import numpy as np
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from copy import deepcopy

class preprosess_Module:
    def __init__(self,df:pd.DataFrame):
        self.df = df
    
    def get_X(self, df:pd.DataFrame, df_tst:pd.DataFrame, features:iter=None):
        '''Make feature vectors from a DataFrame.

        Args:
            df: DataFrame
            features: selected columns
        '''    
        df = deepcopy(df)
        df_tst = deepcopy(df_tst)
        df.dropna(axis=0, subset=['ECLO'], inplace=True)
        df_tst.drop(['ID'], axis=1, inplace=True)
        df = df[df_tst.columns]
        
        df_num = df.select_dtypes(exclude=['object'])
        df_tst_num = df_tst.select_dtypes(exclude=['object'])
        
        missing_df = (df_num.isnull().sum())
        
        all_columns = df_num.columns
        df_num = df_num.drop(all_columns[missing_df > 0], axis=1)
        df_tst_num = df_tst_num.drop(all_columns[missing_df > 0], axis=1)
        
        # 결측치 저리
        df_num = df_num.fillna(df_num.min())
        df_tst_num = df_tst_num.fillna(df_tst_num.min())
        
        df_cat = df.select_dtypes(include=['object'])
        df_cat_tst = df_tst.select_dtypes(include=['object'])
        
        time_pattern = r'(\d{4})-(\d{1,2})-(\d{1,2}) (\d{1,2})' 

        df_cat[['연', '월', '일', '시간']] = df_cat['사고일시'].str.extract(time_pattern)
        df_cat[['연', '월', '일', '시간']] = df_cat[['연', '월', '일', '시간']].apply(pd.to_numeric) # 추출된 문자열을 수치화해줍니다 
        df_cat = df_cat.drop(columns=['사고일시']) # 정보 추출이 완료된 '사고일시' 컬럼은 제거합니다 

        # 해당 과정을 test_x에 대해서도 반복해줍니다 
        df_cat_tst[['연', '월', '일', '시간']] = df_cat_tst['사고일시'].str.extract(time_pattern)
        df_cat_tst[['연', '월', '일', '시간']] = df_cat_tst[['연', '월', '일', '시간']].apply(pd.to_numeric)
        df_cat_tst = df_cat_tst.drop(columns=['사고일시'])

        location_pattern = r'(\S+) (\S+) (\S+)'
        
        df_cat[['도시', '구', '동']] = df_cat['시군구'].str.extract(location_pattern)
        df_cat = df_cat.drop(columns=['시군구'])

        df_cat_tst[['도시', '구', '동']] = df_cat_tst['시군구'].str.extract(location_pattern)
        df_cat_tst = df_cat_tst.drop(columns=['시군구'])

        road_pattern = r'(.+) - (.+)'
        
        df_cat[['도로형태1', '도로형태2']] = df_cat['도로형태'].str.extract(road_pattern)
        df_cat = df_cat.drop(columns=['도로형태'])

        df_cat_tst[['도로형태1', '도로형태2']] = df_cat_tst['도로형태'].str.extract(road_pattern)
        df_cat_tst = df_cat_tst.drop(columns=['도로형태'])

        enc = LabelEncoder()
        df_cat.loc[:,:] = df_cat.loc[:,:].apply(enc.fit_transform)
        df_cat_tst.loc[:,:] = df_cat_tst.loc[:,:].apply(enc.fit_transform)
        
        df_cat.to_csv('encodeing_df.csv')
        #df_Label = enc.fit_transform(df_cat)
        #df_tst_Label = enc.transform(df_cat_tst)
        
        df = np.concatenate([df_num, df_cat], axis=1)
        df_tst = np.concatenate([df_tst_num, df_cat_tst], axis=1)
        
        #print('shape of data:',df.shape, df_tst.shape ,'\nnull count:',np.isnan(df).any(),np.isnan(df_tst).any())
            
        #print('\n입력된 Feature:\n{} \n\nFeature 수: {}\n'.format(features,len(features)))
        # from https://www.kaggle.com/code/alexisbcook/titanic-tutorial
        return df.astype(np.float32), df_tst.astype(np.float32)

    def get_y(self, df:pd.DataFrame):
        '''Make the target from a DataFrame.

        Args:
            df: DataFrame
        '''
        df = deepcopy(df)
        df.dropna(axis=0, subset=['ECLO'], inplace=True)
        df = df['ECLO']
        #to_numpy(dtype=np.float64)
        #print(np.expand_dims(df.to_numpy(dtype=np.float64),axis=1))
        return df.to_numpy(dtype=np.float32)
    
    def __call__(self, df:pd.DataFrame):
        return
    