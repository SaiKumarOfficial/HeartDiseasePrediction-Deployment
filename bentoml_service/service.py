import bentoml
import bentoml.sklearn
import numpy as np                        
import pandas as pd             
from bentoml.io import NumpyNdarray,PandasDataFrame

# load model
model_tag = "sklearn_hdp_model:3xvrz6ysjcj65wg3"

classifier = bentoml.sklearn.load_runner(model_tag,function_name = "predict")
# create a service with the model

service = bentoml.Service(
    "heart_disease_prediction", runners = [classifier]
)

@service.api(input=PandasDataFrame(),output = NumpyNdarray())
def predict(df: pd.DataFrame) ->np.array:
    # predict 
    result = classifier.run(df)
    return np.array(result)


