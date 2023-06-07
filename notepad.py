# Import standard libraries
import logging
import os
from typing import List

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

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

nltk.download("punkt", quiet=True)

def db_exists() -> None:
    engine = create_engine(f"sqlite:///sentences.db", echo=True)
    inspector = inspect(engine)
    
    return "sentences" in inspector.get_table_names()

def faiss_index_exists() -> None:
    return os.path.isfile("sentences.faiss")

metadata = MetaData()
sentences_table = Table("sentences", metadata, Column("sent_idx", Integer, primary_key=True), Column("original_text", String), 
                        Column("clean_text", String), Column("documents", PickleType), Column("embedding", PickleType))

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

    embed = hub.load("./universal-sentence-encoder")

    config = ConfigParser()
    config.read("<ADD PATH>")
    username = config.get("default", "username")
    password = config.get("default", "password")

    data = pd.read_excel("<ADD FILE NAME>", engine="openpyxl")
    unique_ids = list(data["<ADD KEY>"])
    
    <ADD TOKEN, METADATA AND TEXT FUNCTIONS>
    
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
                token = <GET TOKEN>(username, password)
                logging.info(f"New token fetched for batch starting at {count}.")

            document_text = <GET TEXT>(unique_id, token)
            document_metadata = <GET METADATA>(unique_id, token)
            document_name = <PARSE METADATA>

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
                        "document_ids": [unique_id],
                        "embedding": embedding
                    }

                    batch_sent_idx.append(count)
                    seen_sentences[clean_sentence] = count

                    batch_sentences.append({
                        "sent_idx": count,
                        "original_text": raw_sentence,
                        "clean_sentence": clean_sentence,
                        "document_ids": [unique_id],
                        "embedding": embedding
                    })
                else:
                    sentence_index = seen_sentences[clean_sentence]
                    unique_sentences[sentence_index]["document_ids"].append(unique_id)

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

    def semantic_search(query, num_results):
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
                document_ids = result[3]
                original_text = result[1]

                sentence_info = {
                    "text": original_text,
                    "document_ids": document_ids,
                    "similarity_score": float(similarity_score)
                }
                
                results.append(sentence_info)
            else:
                logging.info(f"No result found for index_id {index_id}")

        return results
    
app = FastAPI(
    title="Semantic Similarity Search Demo",
    description="This is a simple API for conducting semantic similarity searches to find language in Executed Agreements. The API utilizes a Faiss index, developed by Facebook AI, and the Universal Sentence Encoder from Google.",
    version="1.0.0"
)

class Search(BaseModel):
    query: str
    num_results: int

@app.get("/health")
def health_check():
    return {"message": "The server is running."}

def process_documents_task(document_ids: List[str]):
    process_documents(document_ids)

@app.post("/process")
async def process_documents(background_tasks: BackgroundTasks, document_ids: List[str]):
    background_tasks.add_task(process_documents_task, document_ids)
    return {"message": f"Processing started for {len(document_ids)} documents."}

@app.post("/search")
def search(request: Search):
    query = request.query
    num_results = request.num_results
    results = semantic_search(query, num_results)
    return {"results": results}

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        session.close()
