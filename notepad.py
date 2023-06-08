# Import standard libraries
import logging
import os
from typing import Any, Dict, List, Tuple, Union

# Import third-party libraries
import faiss
import nltk
import numpy as np
import pandas as pd
import requests
import tensorflow_hub as hub
import uvicorn
from configparser import ConfigParser
from fastapi import FastAPI, BackgroundTasks
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel
from sqlalchemy import create_engine, inspect, select, Table, Column, Integer, MetaData, PickleType, String
from sqlalchemy.orm import sessionmaker

# Initialize logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# Download NLTK data
nltk.download("punkt", quiet=True)

# Define the database schema
metadata = MetaData()
sentences_table = Table("sentences", metadata, Column("sent_idx", Integer, primary_key=True), 
                        Column("original_text", String), Column("clean_text", String), 
                        Column("documents", PickleType), Column("embedding", PickleType))

# Load the Universal Sentence Encoder from disk
embed = hub.load("./universal-sentence-encoder")

# Define the search request model
class Search(BaseModel):
    query: str
    num_results: int
        
# Define the FastAPI app
app = FastAPI(
    title="Semantic Similarity Search Demo",
    description="This is a simple API for conducting semantic similarity searches to find language in Executed Agreements. The API utilizes a Faiss index, developed by Facebook AI, and the Universal Sentence Encoder from Google.",
    version="1.0.0"
)

def db_exists() -> bool:
    """Check if the database exists"""
    engine = create_engine(f"sqlite:///sentences.db", echo=True)
    inspector = inspect(engine)
    
    return "sentences" in inspector.get_table_names()

def faiss_index_exists() -> bool:
    """Check is the Faiss index exists"""
    return os.path.isfile("sentences.faiss")

# If database and index exist, load them
# Otherwise, create new ones
if db_exists() and faiss_index_exists():
    # Connect to existing database
    engine = create_engine(f"sqlite:///sentences.db", echo=True)
    Session = sessionmaker(bind=engine)
    session = Session()
    logging.info("Connected to existing database.")

    # Load Faiss index from disk
    faiss_index = faiss.read_index("sentences.db")
    logging.info("Loaded Faiss index from disk.")
else:
    # Create new database
    engine = create_engine("sqlite:///sentences.db", echo=True)
    metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Create Faiss index
    dim = 512
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index = faiss.IndexIDMap(faiss_index)

    # Load credentials
    config = ConfigParser()
    config.read("H:/jpmDesk/Desktop/credentials.ini")
    username = config.get("default", "username")
    password = config.get("default", "password")
    
    # Load document data
    data = pd.read_excel("executedAgreements5-22-2023.xlsx", engine="openpyxl")
    unique_ids = list(data["UnifiedDocID"])
    
    def get_doclink_token(username: str, password: str) -> str:
        """Retrieve doclink token for given user"""
        url = "https://idag2.jpmorganchase.com/adfs/oauth2/token"
        payload = {
            "client_id": "PC-34963-SID-20429-PROD",
            "grant_type": "password",
            "username": "NAEAST\\" + username,
            "password": password,
            "resource": "JPMC:URI:RS-34963-18863-AWMContentCloud-PROD"
        }
        response = requests.post(url, payload)
        token = response.json()["access_token"]
        return token
    
    def get_document_metadata(unique_id: str, token: str) -> str:
        """Retrieve document metadata"""
        url = f"https://ecm-doclink-services.prod.gaiacloud.jpmchase.net/api/core/v1/app/Scribe/documents/{unique_id}"
        payload = {}
        headers = {
            "Accept": "application/json",
        }
        response = requests.get(url=url, headers=headers, data=payload)
        return response.text
    
    def process_documents(document_ids: List[str]) -> None:
        pass
    
    # Process documents
    count = 0
    
    unique_sentences = {}
    seen_sentences = {}
    
    batch_sentences = []
    batch_sent_idx = []
    batch_embeddings = []
    
    BATCH_SIZE = 10000

    for unique_id in unique_ids:
        try:
            if count % BATCH_SIZE == 0:
                token = get_doclink_token(username, password)
                logging.info(f"New token fetched for batch starting at {count}.")

            document_text = get_document_text(unique_id, token)
            document_metadata = get_document_metadata(unique_id, token)
            document_name = document_metadata["metadata"]["core"]["documentName"]

            raw_sentences = sent_tokenize(document_text)

            for raw_sentence in raw_sentences:
                clean_sentence = raw_sentence.lower()

                if not clean_sentence.strip():
                    continue
            
                if clean_sentence not in seen_sentences:
                    embedding = embed([clean_sentence]).numpy()[0].tolist()
                    batch_embeddings.append(embedding)

                    unique_sentences[count] = {
                        "original_text": raw_sentence,
                        "clean_text": clean_sentence,
                        "documents": [{"id": unique_name, "name": document_name}],
                        "embedding": embedding
                    }

                    batch_sent_idx.append(count)
                    seen_sentences[clean_sentence] = count

                    batch_sentences.append({
                        "sent_idx": count,
                        "original_text": raw_sentence,
                        "clean_sentence": clean_sentence,
                        "documents": [{"id": unique_id, "name": document_name}],
                        "embedding": embedding
                    })
                else:
                    sentence_index = seen_sentences[clean_sentence]
                    unique_sentences[sentence_index]["documents"].({"id": unique_id, "name": document_name})

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

    faiss.write_index(faiss_index, "sentences.faiss")

    def semantic_search(query: str, num_results: int) -> List[Dict[str, Union[str, List[Dict[str, str]], float]]]:
        """Perform a semantic search given a query and the desired number of results"""
        clean_query = query.lower()
        query_embedding = embed([clean_query])

        D, I = faiss_index.search(query_embedding, k=num_results)

        results = []

        for i in range(I.shape[1]):
            index_id =  int(I[0, 1])
            similarity_score = D[0, 1]

            stmt = select(sentences_table).where(sentences_table.c.sent_idx == index_id)
            result = session.execute(stmt).fetchone()

            if result is not None:
                documents = result[3]
                original_text = result[1]

                sentence_info = {
                    "text": original_text,
                    "documents": documents,
                    "similarity_score": float(similarity_score)
                }
                
                results.append(sentence_info)
            else:
                logging.info(f"No result found for index_id {index_id}")

        return results

@app.get("/health")
def health_check() -> Dict[str, str]:
    """Returns a message indicating that the server is running"""
    return {"message": "The server is running."}

def process_documents_task(document_ids: List[str]) -> None:
    """Process the specified documents"""
    process_documents(document_ids)

@app.post("/process")
async def process_documents(background_tasks: BackgroundTasks, document_ids: List[str]) -> Dict[str, str]:
    """Start a background task to process the specified documents"""
    background_tasks.add_task(process_documents_task, document_ids)
    logging.info(f"Processing started for {len(document_ids)} documents.")
    
    return {"message": f"Processing started for {len(document_ids)} documents."}

@app.post("/search")
def search(request: Search) -> Dict[str, Any]:
    """Perform a semantic search and return the results"""
    query = request.query
    num_results = request.num_results
    results = semantic_search(query, num_results)
    
    return {"results": results}

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        session.close()
