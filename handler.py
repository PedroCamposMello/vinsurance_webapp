import os
import pickle
import pandas as pd
from flask             import Flask, request, Response
from apps.web_app.custom_packs.v_insurance      import V_insurance

# loading model
model = pickle.load( open( 'exports/cicle_products/model_xgb.pkl', 'rb') )

# initialize API
app = Flask( __name__ )

@app.route( '/vinsurance/predict', methods=['POST'] )
def V_insurance_predict():
    test_json = request.get_json()
   
    if test_json: # there is data
        if isinstance( test_json, dict ): # unique example
            df_00 = pd.DataFrame( test_json, index=[0] )
            
        else: # multiple example
            df_00 = pd.DataFrame( test_json, columns=test_json[0].keys() )
            
        # Instantiate Rossmann class
        pipeline = V_insurance()
        
        # data wrangling
        df_01 = pipeline.data_cleaning( df_00 )
        df_05 = pipeline.data_preparation( df_01 )
        df_06 = pipeline.feature_selection( df_05 )

        # prediction
        df_response = pipeline.get_prediction( model, df_00, df_06 )

        return df_response
          
    else:
        return Response( '{}', status=200, mimetype='application/json' )

if __name__ == '__main__':
    port = os.environ.get( 'PORT', 5000 )
    app.run( host='0.0.0.0', port=port )