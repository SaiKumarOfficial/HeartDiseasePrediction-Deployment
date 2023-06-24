# HeartDiseasePrediction-Deployment

This repository contains a machine learning model for predicting heart disease using various features and data.The model can be used to predict the likelihood of a person having heart disease based on their input.


## Tools I have used

- Python
- DVC (Data Version Control)
- BentoMl
## Why we use DVC and bentoml
BentoML: We use BentoML for model deployment because it allows us to easily package and serve our trained model as a REST API, enabling seamless integration into applications.

DVC: We use DVC (Data Version Control) to version our data and model files, ensuring reproducibility and efficient management of large files, enabling collaboration and easy tracking of changes. 

## Steps to run
1. Clone this repository to your local machine.
2. Ensure you have Python (>=3.6) installed.
3. Install the necessary Python packages by  running the following command:
```bash
pip install -r requirements.txt
```
4. To version the data and model files, follow these steps:
    - Initialize DVC using the command:
    ```bash
    dvc init
    ```
    - Add files to remote storage
    ```bash
    dvc add data_given/heart.csv
    ```
    - Define the pipeline with dvc.yaml file
    ```bash
    dvc repro
    ```
    - I had created reports. To look at some of the model metrics, run
    ```bash
    dvc metrics show
    ```
    - Let's see the confusion matrix
    ```bash
    dvc plots show cm.csv --template confusion -x chd -y Predicted
    ```
5. Now, you can do experiments by changing some of the parameters in params.yam and rerun the dvc pipeline.To rerun it , again run the above following commands.
6. To see the model improvement
```bash
dvc metrics diff
```
## Deploy Ml model using Bentoml:
1. First, save the model in bentoml storage
```bash
python savemodelToBento.py
```
2. To Create a bentoml servic,go to bentoml_service directory
```bash
cd bentoml_service
```
```bash
bentoml serve service:service --reload
```
3. After running the above command, to send the request to bentoml, run serviceRequest.py file in another command prompt
```bash
python serviceRequest.py
```
4. To build model and service into a bento
```bash
bentoml build
```
5. To serve the model through bento
```bash
bentoml serve <service-tag>:latest --production
```
6. Again run serviceRequest.py file to send the request
7. Dockerise the bento, it will automatically create one docker image
```bash
bentoml containerize <service-tag>
``` 
8. Run bento service via Docker
```bash
docker run -p 3000:3000 <docker-image-name>
```
Again send the request by running serviceRequest.py file.
