# Import standard libraries
import logging
import os
import re
import string
from typing import Any, Dict, List, Union

# Import third-party libraries
import faiss
import nltk
import numpy as np
import pandas as pd
import requests
import tensorflow_hub as hub
import uvicorn
from configparser import ConfigParser
from fastapi import FastAPI
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel
from sqlalchemy import create_engine, inspect, select, Table, Column, Integer, MetaData, PickleType, String
from sqlalchemy.orm import sessionmaker

# Initialize logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# Define use case
USE_CASE = "executed-agreements"

# Download NLTK data
def download_nltk_data():
    """Download necessary NLTK data"""
    nltk_data_dir = os.path.expanduser("~/nltk-data")
    punkt_path = os.path.join(nltk_data_dir, "tokenizers/punkt")

    if not os.path.exists(punkt_path):
        nltk.download("punkt", quiet=True)

download_nltk_data()

# Define the database schema
metadata = MetaData()
sentences_table = Table(f"{USE_CASE}", metadata, Column("sent_idx", Integer, primary_key=True), 
                        Column("text", String), Column("document", PickleType))

# Load Universal Sentence Encoder
embed = hub.load("./universal-sentence-encoder")

# Define the search request model
class Search(BaseModel):
    query: str
    num_results: int = 10
    context: int = 0

# Create the FastAPI app
app = FastAPI(
    title="Semantic Similarity Search Demo",
    description="This is a simple API for conducting semantic similarity searches to find language in Executed Agreements.",
    version="1.0.0"
)

def db_exists() -> bool:
    """Check if the database exists"""
    engine = create_engine(f"sqlite:///{USE_CASE}.db", echo=True)
    inspector = inspect(engine)
    return USE_CASE in inspector.get_table_names()

def faiss_index_exists() -> bool:
    """Check if the index exists"""
    return os.path.isfile(f"{USE_CASE}.faiss")

def get_access_token(username: str, password: str) -> str:
    """Retrieve doclink token for given user"""
    url = "<URL>"
    payload = {
        "client_id": "<CLIENT_ID>",
        "grant_type": "password",
        "username": "<DOMAIN>" + username,
        "password": password,
        "resource": "<RESOURCE>"
    }
    response = requests.post(url, payload)
    token = response.json()["access_token"]    
    return token

def get_document_metadata(unique_id: str, token: str) -> str:
    """Retrieve document metadata"""
    url = "<URL>"
    payload = {}
    headers = {
        "Accept": "application/json",
        "Authorization": "Bearer" + token
    }
    response = requests.get(url=url, headers=headers, data=payload)
    return response.json()

def get_document_text(unique_id, token) -> str:
    """Retrieve document text"""
    url = "<URL>"
    payload = {}
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer" + token
    }
    response = requests.get(url=url, headers=headers, data=payload)
    return response.text

def preprocess_text(text: str) -> str:
    """Clean and normalize sentences"""
    text = re.sub(r'[""\"]', '', text)
    text = re.sub(r'\r?\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.lower()

# Check to see if the database and index exist
database_exists = db_exists()
index_exists = faiss_index_exists()

# Connect to existing database or create new one
if database_exists:
    engine = create_engine(f"sqlite:///{USE_CASE}.db", echo=True)
    Session = sessionmaker(bind=engine)
    session = Session()
    logging.info("Connected to existing database.")
else:
    engine = create_engine(f"sqlite:///{USE_CASE}.db", echo=True)
    metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

# Load existing index or create new one
if index_exists:
    faiss_index = faiss.read_index(f"{USE_CASE}.faiss")
    logging.info("Loaded Faiss index from disk.")
else:
    dim = 512
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index = faiss.IndexIDMap(faiss_index)

# Only fetch document data if the database or index don't exist
if not database_exists or not index_exists:
    # Load credentials
    config = ConfigParser()
    config.read("H:/jpmDesk/Desktop/credentials.ini")
    username = config.get("default", "username")
    password = config.get("default", "password")

    # Load document data
    data = pd.read_excel("<FILEPATH>")
    unique_ids = list(data["UnifiedDocID"])
    
    # Process documents
    count = 0

    batch_sentences = []
    batch_sent_idx = []
    batch_embeddings = []

    BATCH_SIZE = 10000

    token = get_doclink_token(username, password)

    for unique_id in unique_ids:
        try:
            if count % BATCH_SIZE == 0 and count != 0:
                token = get_access_token(username, password)
                logging.info(f"New token fetched for batch starting at {count}.")

            document_text = get_document_text(unique_id, token)
            document_metadata = get_document_metadata(unique_id, token)
            document_name = document_metadata["metadata"]["core"]["documentName"]

            raw_sentences = sent_tokenize(document_text)

            for raw_sentence in raw_sentences:
                clean_sentence = preprocess_text(raw_sentence)

                if not clean_sentence.strip():
                    continue

                embedding = embed([clean_sentence]).numpy()[0].tolist()
                batch_embeddings.append(embedding)

                batch_sentences.append({
                    "sent_idx": count,
                    "text": raw_sentence,
                    "document": {"id": unique_id, "name": document_name}
                })

                batch_sent_idx.append(count)
                count += 1

                if count % BATCH_SIZE == 0:
                    with session.begin():
                        session.execute(sentences_table.insert(), batch_sentences)
                        faiss_index.add_with_ids(np.array(batch_embeddings), np.array(batch_sent_idx))

                    batch_sentences = []
                    batch_sent_idx = []
                    batch_embeddings = []

                    logging.info(f"Loaded {count} sentences.")
        except Exception as e:
            logging.error(f"Error processing document with id {unique_id}: {str(e)}")

    if batch_sentences:
        with session.begin():
            session.execute(sentences_table.insert(), batch_sentences)
            faiss_index.add_with_ids(np.array(batch_embeddings), np.array(batch_sent_idx))                        

    faiss.write_index(faiss_index, f"{USE_CASE}.faiss")
    logging.info("Finished processing documents.")

def semantic_search(query: str, num_results: int = 10, context: int = 0) -> List[Dict[str, Union[str, List[Dict[str, str]], float]]]:
    """Perform a semantic search given a query and the desired number of results"""
    clean_query = query.lower()
    query_embedding = embed([clean_query])

    D, I = faiss_index.search(query_embedding, k=num_results)
    
    results = []

    for i in range(I.shape[1]):
        index_id =  int(I[0, i])
        similarity_score = D[0, i]

        stmt = select(sentences_table).where(sentences_table.c.sent_idx == index_id)
        result = session.execute(stmt).fetchone()

        if result is not None:
            document = result[2]
            text = result[1]
            document_id = document["id"]

            if context > 0:
                # Fetch sentences before matched sentence to provide context
                stmt_before = select(sentences_table).where(
                    sentences_table.c.sent_idx.between(index_id - context, index_id - 1)
                )
                result_before = session.execute(stmt_before).fetchall()
                result_before = [r for r in result_before if r[2]["id"] == document_id]

                # Fetch sentences after matched sentence to provide context
                stmt_after = select(sentences_table).where(
                    sentences_table.c.sent_idx.between(index_id + 1, index_id + context)
                )                
                result_after = session.execute(stmt_after).fetchall()
                result_after = [r for r in result_after if r[2]["id"] == document_id]

                context_sentences = [r[1] for r in result_before] + [text] + [r[1] for r in result_after]
                context_sentences = " ".join(context_sentences)
            else:
                context_sentences = text

            sentence_info = {
                "text": context_sentences,
                "document": document,
                "similarity_score": float(similarity_score)
            }
            results.append(sentence_info)
        else:
            logging.info(f"No result found for index_id {index_id}")

    return results

@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Returns a message indicating that the server is running"""
    return {"message": "The server is running."}

@app.post("/search")
def search(request: Search) -> Dict[str, Any]:
    """Perform a semantic search and return the results"""
    query = request.query
    num_results = request.num_results
    context = request.context
    results = semantic_search(query, num_results, context) 
    return {"results": results}

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        session.close()
