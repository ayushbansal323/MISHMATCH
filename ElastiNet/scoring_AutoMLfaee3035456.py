# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import pickle
import numpy as np
import pandas as pd
import azureml.train.automl
from sklearn.externals import joblib
from azureml.core.model import Model

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame(data=[{'Column1': '1970-01-01T00:00:00.000Z', 'Column2': 14.5, 'Column4': 17690892.42857143, 'Column5': 75855.9285714286, 'Column6': 6349730.5, 'Column7': 199714.3214285714, 'Column8': 207294.0714285714, 'Column9': 106157.75, 'Column10': 369996844.53571427, 'Column11': 1910144.9642857143, 'Column12': 22.3571428571, 'Column13': 25.4285714286, 'Column14': 61.1785714286, 'Column15': 53.4642857143, 'Column16': 0.4865860714, 'Column17': 8.5916785714, 'Column18': 0.0629722143, 'Column19': 5522.5357142857, 'Column20': 43.1921428571, 'Column21': 51.965, 'Column22': 8.3394285714, 'Column23': 3.7442857143, 'Column24': 2.3456071429, 'Column25': 1.5301785714, 'Column26': 546619407.6071428, 'Column27': 29.0971428571, 'Column28': 2.5964142857, 'Column29': 0.2433457143, 'Column30': 39.7139285714, 'Column31': 51.1303571429, 'Column32': 29.8110714286, 'Column33': 107.3414285714, 'Column34': 38.6139285714, 'Column35': 45.5846428571, 'Column36': 70.8157142857, 'Column37': 7156284.607142857, 'Column38': 511652.4642857143, 'Column39': 0.2338071429, 'Column40': 39.5778571429}], columns=['Column1', 'Column2', 'Column4', 'Column5', 'Column6', 'Column7', 'Column8', 'Column9', 'Column10', 'Column11', 'Column12', 'Column13', 'Column14', 'Column15', 'Column16', 'Column17', 'Column18', 'Column19', 'Column20', 'Column21', 'Column22', 'Column23', 'Column24', 'Column25', 'Column26', 'Column27', 'Column28', 'Column29', 'Column30', 'Column31', 'Column32', 'Column33', 'Column34', 'Column35', 'Column36', 'Column37', 'Column38', 'Column39', 'Column40'])


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = Model.get_model_path(model_name = 'AutoMLfaee3035456')
    model = joblib.load(model_path)


@input_schema('data', PandasParameterType(input_sample, enforce_shape=False))
def run(data):
    try:
        y_query = None
        if 'y_query' in data.columns:
            y_query = data.pop('y_query').values
        result = model.forecast(data, y_query)
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})

    forecast_as_list = result[0].tolist()
    index_as_df = result[1].index.to_frame().reset_index(drop=True)
    
    return json.dumps({"forecast": forecast_as_list,   # return the minimum over the wire: 
                       "index": json.loads(index_as_df.to_json(orient='records'))  # no forecast and its featurized values
                      })
