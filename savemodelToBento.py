import bentoml 
from pathlib import Path
import joblib
import os


dir_loc = r"C:\\Users\\Sheela Sai kumar\\Documents\\GitHub actions\\HeartDiseasePrediction-Deployment\\saved_models"


def save_model_into_bento(model_file_path: Path) -> None:
    # Loads a keras model from disk and saves it to BentoMl """

    model = joblib.load(model_file_path)
    bento_model = bentoml.sklearn.save_model("sklearn_hdp_model",model)
    print(f"Bento model tag = {bento_model.tag}") 

if __name__ == "__main__":
    for file_name in os.listdir(dir_loc):
        if file_name==".gitignore":
            continue
        file_path = os.path.join(dir_loc, file_name)
        save_model_into_bento(Path(file_path))