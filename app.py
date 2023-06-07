# Import standard libraries
import logging
import os

# Import third-party libraries
import faiss
import nltk
import numpy as np
import pandas as pd
import requests
import tensorflow_hub as hub
from fastapi import FastAPI
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel
from sqlalchemy import create_engine, select, Table, Column, Integer, String, MetaData, PickleType
from sqlalchemy.orm import sessionmaker

nltk.download("punkt", quiet=True)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)








# Utility functions
def preprocess_text(text):
    logging.info("Preprocessing text.")
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]
    words = [word for word in words if word not in stopwords_en]
    words = [stemmer.stem(word) for word in words]
    return " ".join(words)

# Load Universal Sentence Encoder
try:
    embed = hub.load("./universal-sentence-encoder")
    logging.info("Sentence encoder loaded from local 'models' directory.")
except Exception as e:
    logging.warning("Could not load the model locally. Error: {}".format(e))
    logging.warning("Attempting to download model from TensorFlow Hub...")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    logging.info("Sentence encoder downloaded from TensorFlow Hub.")

# Initialize the indexes and documents dictionaries
indexes = {}
documents = {}

def create_index_from_folder(dataset_name, directory_path, preprocess=True):
    index_filepath = f"{dataset_name}_index.faiss"

    if not os.path.exists(index_filepath):
        sentences_dataset = []
        documents_dataset = []

        for filename in os.listdir(directory_path):
            if os.path.splitext(filename)[1] == ".txt":
                with open(os.path.join(directory_path, filename), 'r') as f:
                    data = f.readlines()

                logging.info(f"{filename} data fetched and indexed.")

                for i, document in enumerate(data):
                    if preprocess:
                        document = preprocess_text(document)
                    document_sentences = sent_tokenize(document)
                    sentences_dataset.extend(document_sentences)
                    documents_dataset.extend([{'filename': filename, 'content': document}] * len(document_sentences))

        # Vectorize sentences and create Faiss index
        sentence_vectors_dataset = embed(sentences_dataset).numpy()
        dimension_dataset = sentence_vectors_dataset.shape[1]
        index_dataset = faiss.IndexFlatL2(dimension_dataset)
        index_dataset.add(sentence_vectors_dataset)

        # Save the index to disk
        faiss.write_index(index_dataset, index_filepath)

    else:
        index_dataset = faiss.read_index(index_filepath)

    indexes[dataset_name] = index_dataset
    documents[dataset_name] = documents_dataset

    return f"{dataset_name} index created"

# Create FastAPI app
app = FastAPI(
    title="Semantic Search API", 
    description="This is a simple API for conducting semantic searches that utilizes Faiss and the Universal Sentence Encoder.", 
    version="1.0.0"
)
logging.info("FastAPI app created.")

class Index(BaseModel):
    name: str
    folder_path: str

class Search(BaseModel):
    query: str
    index: str

@app.get("/health")
def health_check():
    return {"message": "The server is running"}

@app.get("/indexes")
def list_indexes():
    return {"indexes": list(indexes.keys())}

@app.post("/create_index")
def create_index(index: Index):
    dataset_name = index.name
    folder_path = index.folder_path

    create_index_from_folder(dataset_name, folder_path)

    return {"message": f"Index {dataset_name} created successfully"}

@app.post("/search")
def search(search: Search):
    query = search.query
    index_name = search.index

    query_vector = embed([query]).numpy()
    D, I = indexes[index_name].search(query_vector, k=5)

    results = []
    for i in range(I.shape[1]):
        document_id = I[0][i]
        document_info = documents[index_name][document_id]
        results.append({
            "document": document_info['content'],
            "filename": document_info['filename'],
            "score": D[0][i],
        })

    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
