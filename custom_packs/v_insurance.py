import pickle
import inflection
import pandas as pd
import numpy as np

class V_insurance( object ):
    def __init__( self ):
        self.home_path='' 
        self.tranf_age                      = pickle.load( open( self.home_path + 'exports/cicle_products/tranf_age.pkl', 'rb') )
        self.tranf_annual_premium           = pickle.load( open( self.home_path + 'exports/cicle_products/tranf_annual_premium.pkl', 'rb') )
        self.tranf_vintage                  = pickle.load( open( self.home_path + 'exports/cicle_products/tranf_vintage.pkl', 'rb') )
        self.tranf_policy_sales_channel     = pd.read_pickle( self.home_path + 'exports/cicle_products/tranf_policy_sales_channel.pkl', compression="gzip") 
        self.tranf_region_code              = pd.read_pickle( self.home_path + 'exports/cicle_products/tranf_region_code.pkl', compression="gzip") 

    def data_cleaning( self, df_01 ):
        cols_old = ['id', 'Gender', 'Age', 'Driving_License', 'Region_Code',
                    'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium',
                    'Policy_Sales_Channel', 'Vintage']

        snakecase = lambda x: inflection.underscore( x )

        cols_new = list( map( snakecase, cols_old ) )

        # Renomeando colunas:
        df_01.columns = cols_new

        # Transformações:

        # Categóricos -> Object:
        col_select = ['id',
                    'gender',
                    'driving_license',
                    'region_code',
                    'previously_insured',
                    'vehicle_age',
                    'vehicle_damage',
                    'policy_sales_channel'
                    ]

        df_01[col_select] = df_01[col_select].astype(str)

        return df_01 

    def data_preparation( self, df_05 ):
        # annual_premium:
        df_05['annual_premium'] = self.tranf_annual_premium.transform(df_05[['annual_premium']].values)

        # age
        df_05['age'] = self.tranf_age.transform(df_05[['age']].values)

        # vintage
        df_05['vintage'] = self.tranf_vintage.transform(df_05[['vintage']].values)

        # id - Já está no formato de Label Encoding
        df_05['id'] = df_05['id'].astype(int)

        # gender - Label Encoding
        df_05['gender'] = df_05['gender'].apply(lambda x: 1 if x=='Male' else 0)

        # driving_license - Já está no formato de Label Encoding
        df_05['driving_license'] = df_05['driving_license'].astype(int)

        # region_code - Target Encoding
        df_05['region_code'] = df_05['region_code'].astype(float)
        df_05['region_code'] = df_05['region_code'].map(self.tranf_region_code)

        # previously_insured - Já está no formato de Label Encoding
        df_05['previously_insured'] = df_05['previously_insured'].astype(int)

        # vehicle_age - Ordinal Encoding
        dict_categoria = {'< 1 Year': 1,  '1-2 Year': 2, '> 2 Years': 3}
        df_05['vehicle_age'] = df_05['vehicle_age'].map( dict_categoria )

        # vehicle_damage - Label Encoding
        df_05['vehicle_damage'] = df_05['vehicle_damage'].apply(lambda x: 1 if x=='Yes' else 0)

        # policy_sales_channel - Frenquency Encoding
        df_05['policy_sales_channel'] = df_05['policy_sales_channel'].astype(float)
        df_05['policy_sales_channel'].map(self.tranf_policy_sales_channel)

        return df_05

    def feature_selection( self, df_06 ):

        cols_selected = [
                        'vintage',
                        'annual_premium',
                        'age',
                        'region_code',
                        'vehicle_damage',
                        'policy_sales_channel',
                        'previously_insured'
                        ]
        
        return df_06[cols_selected]
    
    def get_prediction( self, model, original_data, test_data ):
        # model prediction
        pred = model.predict_proba( test_data )
        
        # join prediction into original data
        original_data['prediction'] = pred
        
        return original_data.to_json( orient='records', date_format='iso' )
    
