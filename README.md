# BFH SAI3 – FS25
This repository holds  the code for the SAI3 Module @ https://www.bfh.ch/. It explores a simple RAG implementation using python.
The main Product is our Chatbot which lets you search through our vectorized recipee database. It includes prompts to enrich result's and have a lively chat with the end user.It is authored by:

- Samuel Nussbaumer
- Fahrenbruch Timo 
- Kunz Eric Alexander
- Niggli Dominik
- Weingart Brandon Suriyan

# General Information / System Architecture.
Our Application acts as a search engine/chatbot for a large recipe database. 
To achieve this we use parquet database files from a dataset found on huggingface. We then use pandas, numpy & chromadb to
vectorize the dataset and store it in a db. As it can take quite a time to vectorize we also added a `vector_store_checkpoint.json`within the /database folder file to keep track of what has already been indexed.
Afterwards, we query it using langchain, llama3.2 and mxbai-embed-large. 
We implement a prompt enhancer to gather more usefull result. But it also features a raw mode for querying directly to the database.

The implementation is built open these core dependencies:

- numpy
- pandas
- chroma db
- langchain
- llama3.2
- mxbai-embed-large
- https://huggingface.co/datasets/corbt/all-recipes
- flask

# Install guidelines
A sample vectorised database is available so you don't have to do that yourself. For the best experience please run it locally as the docker image is quite resource heavy 😅.

## Requirements
- Docker (if you only run it with docker the next things don't apply to you)
- ollama
- llama3.2
- mxbai-embed-large

Pro-Tip: Please assign sufficient processing power to your docker engine as it can get quite processing-heavy and is not optimized. Or run it locally instead...

## Getting Started
**Please download the database.zip** and unpack it in the root of this repository: https://bernerfachhochschule-my.sharepoint.com/:u:/g/personal/nusss8_bfh_ch/EYesewYJn9hFnotQYC9g3zcBEVc_6M1C1QFz3uTmKhXkBA?e=RVa2cn

**Please note that this is only tested on macos. Other systems are not supported!**


> We decided to switch to a setup where docker connects to the local ollama instance. This is due to the reason that inside the docker container llama3.2 is not able to access the gpu and therefore is really, really slow (or we built something wrong, sorry). The initial Dockerfile can be found in `src/legacy/Dockerfile`

Also make sure that you have ollama, llama3.2 and mxbai-embed-large running locally.
1. install ollama `brew install ollama`
2. start ollama background service `brew services start ollama`
3. pull llama3.2 `ollama pull llama3.2`
4. pull the embed model `ollama pull mxbai-embed-large`

Afterward, please run the following commands to get the app running:
1. `docker build -t sai2-fs25-group2 .`
2. `docker run -p 1337:1337 sai2-fs25-group2`

Navigate to http://localhost:1337 and enjoy the app!
Attention: If possible try to run the system locally due to performance issues

## Running things locally
Please note that this is only tested on macos. Other systems are not supported!

It is also important to have llama3.2 and mxbai-embed-large running locally. Please run the following if not already done:

1. install ollama `brew install ollama`
2. start ollama background service `brew services start ollama`
3. pull llama3.2 `ollama pull llama3.2`
4. pull the embed model `ollama pull mxbai-embed-large`

Application install steps:
1. install 3.11.9 using pyenv (if not installed already) `pyenv install 3.11.9`
2. create new pyenv `pyenv virtualenv 3.11.9 sai3-fs25-group2`
3. activate pyenv `pyenv activate sai3-fs25-group2`
4. install requirements `pip install -r requirements.txt`
5. fetch data by running `./fetch_data.sh`
6. run webserver `python src/webserver.py`

Navigate to http://localhost:1337 and enjoy the app!

# Favorite Search Terms

- funky recipe -> Recipe For A Happy Day
- Big Mac Sauce
- Swiss Vegetable Medley
- Recipe For Life

# Learinings

- Hallucination is really strange...
- Python dependency management is _interesting_
- Python is a cool language though

