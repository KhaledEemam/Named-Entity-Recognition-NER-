NER Project
==============================

This project aims to detect the names of the entities included within a given text. I used two approaches in this project. The first is to try to train a BI-LSTM-CRF model, and the other one was fine-tuning the BertModel for this task which achieved 96% on f1_score evaluation metric. After observing both results, I decided to take the fine-tuned Bert model to the next step and create a FAST API to take the text input from the user and return the tags for each word in the input. Also I created a Dockerfile to facilitate the installation of project dependencies. The available tags the model was trained on are 'geo', 'tim', 'org', 'per', 'gpe', 'O', which represent "Geographical Entity", "Time", "Organization", "Person", "Geo-Political Entity", "Other" accordingly.

Technologies stack :
==============================
* Python
* Pytorch
* FastAPI
* Docker


Project Structure
==============================
```bash
.
├── README.md
└── src
    ├── Dockerfile
    ├── Notebooks
    │   ├── 01-exploring_data.ipynb
    │   └── __init__.py
    ├── assets
    │   ├── data
    │   │   ├── preprocessed
    │   │   │   └── preprocessed_NER_dataset.csv
    │   │   └── raw
    │   │       └── original_NER_dataset.csv
    │   ├── postman collections
    │   │   └── ner-app.postman_collection.json
    │   └── trained_models
    │       ├── best_model.bin
    │       └── tokenizer.bin
    ├── controllers
    │   ├── BaseController.py
    │   ├── DataController.py
    │   └── __init__.py
    ├── helpers
    │   ├── __init__.py
    │   └── config.py
    ├── main.py
    ├── models
    │   ├── __init__.py
    │   ├── bert_model.py
    │   └── enums
    │       ├── ResponseEnums.py
    │       └── __init__.py
    ├── requirements.txt
    ├── routes
    │   ├── __init__.py
    │   ├── base.py
    │   └── labels_response.py
    ├── tasks
    │   ├── __init__.py
    │   ├── build_dataset.py
    │   ├── evaluate.py
    │   ├── get_loader.py
    │   ├── inference.py
    │   ├── preprocess_data.py
    │   └── train.py
    └── train_and_evaluate.py
```


Requirements
==============================
- Python 3.8 or later

## Install Python using MiniConda

1) Download and install MiniConda from [here](https://docs.anaconda.com/free/miniconda/#quick-command-line-install)
2) Create a new environment using the following command:
```bash
$ conda create -n mini-rag python=3.8
```
3) Activate the environment:
```bash
$ conda activate mini-rag
```
 Make sure that you have installed CUDA and cuDNN. Also make sure that you have Docker installed on your device if you are going to folloe the Docker installation instructions.

# **Installation**

## 1- *Manual Configuration*

### Install the required packages

```bash
$ pip install -r requirements.txt
```

### Setup the environment variables

```bash
$ cp .env.example .env
```

Set your environment variables in the `.env` file. Like RAW_DATA_PATH & PREPROCESSED_DATA_PATH.

## Change directory to src

```bash
$ cd src
```

### Train your model

```bash
$ python main.py
```

### Run the FastAPI server

```bash
$ uvicorn main:app --reload --port 5000
```

### Import the postman collection 
- open Postman
- Import the provided collection from "assets/postman collections/ner-app.postman_collection.json"
- Navigate to the get_labels API
- Enter your input text

### Output Sample

```bash
Input : "Steve Jobs co-founded Apple Inc. in California."
output : 
{
    "Valid": true,
    "Message": {
        "per": [
            "steve",
            "jobs"
        ],
        "O": [
            "co",
            "-",
            "founded",
            "in",
            "."
        ],
        "org": [
            "apple",
            "inc",
            "."
        ],
        "geo": [
            "california"
        ]
    }
}
```
--------
## 2- *Docker Configuration*

## Change directory to src

```bash
$ cd src
```

### Build the image

```bash
$ docker build -t train .
```

### Run the image

```bash
$ docker run -it -d --gpus all -v /absolute/path/for/src/folder/on/your/device:/ner --name c1 -p 5000:4000 train
```
replace absolute/path/for/src/folder/on/your/device with your src path

### Switch to the container terminal

```bash
$ docker exec -t c1 /bin/sh
```

### Train your own model

```bash
$ python3 train_and_evaluate.py
```

### Run fastapi server

```bash
$ uvicorn main:app --reload port 4000
```
### Import the postman collection 
- open Postman
- Import the provided collection from "assets/postman collections/ner-app.postman_collection.json"
- Navigate to the get_labels API
- Enter your input text