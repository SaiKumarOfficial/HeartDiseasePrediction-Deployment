import yaml
import pandas as pd     
import argparse
import os        

def read_params(config_name):
    # CONFIG_PATH = "../"
    # os.path.join(CONFIG_PATH, config_name)
    with open(config_name) as file:
        config = yaml.safe_load(file)
    return config 

def get_data(config_path):
    config = read_params(config_path)
    data_path = config["data_source"]
    df = pd.read_csv(data_path)
    df = pd.get_dummies(df,drop_first = True)
    return df

if __name__=="__main__":
    args  = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    # config_path = "params.yaml"
    data = get_data(config_path = parsed_args.config )

