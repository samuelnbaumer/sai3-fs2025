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

# Install guidelines
A sample vectorised database is available so you don't have to do that yourself. If you want to test the `vecctor.py`please refer to the steps mentioned under `Additional Steps`

## Requirements
- Docker

Pro-Tip: Please assign sufficient processing power to your docker engine as it can get quite processing-heavy.

## Getting Started
Please download the database.zip and unpack it in the root of this repository: https://bernerfachhochschule-my.sharepoint.com/:u:/g/personal/nusss8_bfh_ch/EYesewYJn9hFnotQYC9g3zcBEVc_6M1C1QFz3uTmKhXkBA?e=RVa2cn

Afterward, please run the following commands to get the app running:
1. `docker build -t sai2-fs25-group2 .`
2. `docker run -it sai2-fs25-group2`

## Additional Steps



# Favorite Search Terms

- Tired Wife'S Supper
- `Big Mac' Sauce
- Ole'! Macaroni & Creamy Cheese Sauce!

# Learinings

- Hallucination is really strange...
- Python dependency management is interesting
- Python is a cool language though

## Tech Stack

- vectorising -> langchain_ollama
- document extraction -> pandas
- Chroma DB https://www.trychroma.com/
- llama -> llama3.2
  ~~- frontend -> https://github.com/open-webui/open-webui~~
