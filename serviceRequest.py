from typing import Tuple
import json
import pandas as pd 
import numpy as np
import requests
from  src.train_and_evaluate import train_and_evaluate
from src.get_data import read_params

SERVICE_URL = "http://localhost:3000/predict"

config_path = "params.yaml"

# def sample_random_mnist_data_point() -> Tuple[np.ndarray, np.ndarray]:
#     _, _, test_images, test_labels = prepare_mnist_training_data()
#     random_index = np.random.randint(0, len(test_images))
#     random_test_image = test_images[random_index]
#     random_test_image = np.expand_dims(random_test_image, 0)
#     return random_test_image, test_labels[random_index]
def sample_heartdisease_data(config_path) -> Tuple[np.ndarray,np.ndarray]:
    config = read_params(config_path)
    test_path = config['split_data']['test_path']
    target = config['base']['target_col']
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(target, axis=1)
    y_test = test_df[target]
    return X_test.values, y_test.values

def make_request_to_bento_service(
    service_url: str, input_array: np.ndarray
) -> str:
    serialized_input_data = json.dumps(input_array.tolist())
    response = requests.post(
        service_url,
        data=serialized_input_data,
        headers={"content-type": "application/json"}
    )
    return response.text


def main():
    input_data, expected_output = sample_heartdisease_data(config_path)
    prediction = make_request_to_bento_service(SERVICE_URL, input_data)
    dic  = {"actual": expected_output,"predicted": prediction}
    df = pd.DataFrame(dic)
    converted_list = json.loads(df['predicted'][0])
    for i,ele in enumerate(converted_list):
        df['predicted'][i] = ele
    df.to_csv("results.csv", index=False)
    print(f"Prediction: {prediction}")
    print(f"Expected output: {expected_output}")


if __name__ == "__main__":
    main()