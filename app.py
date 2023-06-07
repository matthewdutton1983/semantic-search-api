# Import standard libraries
import glob
import logging
import os
import uuid

# Import third-party libraries
import faiss
import nltk
import numpy as np
import pandas as pd
import requests
import tensorflow_hub as hub
import uvicorn
from fastapi import FastAPI
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel
from sqlalchemy import create_engine, select, Table, Column, Integer, String, MetaData, PickleType
from sqlalchemy.orm import sessionmaker

nltk.download("punkt", quiet=True)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

try:
    embed = hub.load("./universal-sentence-encoder")
    logging.info("Sentence encoder loaded from local 'models' directory.")
except Exception as e:
    logging.warning("Could not load the model locally. Error: {}".format(e))
    logging.info("Attempting to download model from TensorFlow Hub...")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    logging.info("Sentence encoder downloaded from TensorFlow Hub.")

dim = 512
faiss_index = faiss.IndexFlatL2(512)
faiss_index = faiss.IndexIDMap(faiss_index)

engine = create_engine("sqlite:///sentences.db", echo=False)
metadata = MetaData()
sentences_table("sentences", metadata, Column("sent_idx", Integer, primary_key=True), Column("original_text", String), Column("clean_text", String), Column("document_ids", PickleType), Column("embedding", PickleType))
metadata.create_all(engine)

Session = sessionmaker(bind=engine)
Session = Session()  

# Process documents
def process_text(text):
    return text.lower()

count = 0
unique_sentences = {}
seen_sentences = {}
sent_indexes = []
embeddings = []

folder_path = "./documents/"
documents = glob.glob(folder_path + "*")

BATCH_SIZE = 250

for document in documents:
    try:
        unique_id = uuid.uuid4()

        with open(document, "r") as f:
            text = f.read()
            name = os.path.basename(document)

        raw_sentences = sent_tokenize(text)

        for raw_sentence in raw_sentences:
            clean_sentence = process_text(text)

            if not clean_sentence.strip():
                continue

            if clean_sentence not in seen_sentences:
                embedding = embed([clean_sentence]).numpy()[0].tolist()
                embeddings.append(embedding)

                unique_sentences[count] = {
                    "original_text": raw_sentence,
                    "clean_text": clean_sentence,
                    "document_ids: [unique_id],
                    "embedding": embedding
                }

                sent_indexes.append(count)
                seen_sentences[clean_sentence] = count

                stmt = sentences_table.insert().values(
                    sent_idx = count,
                    original_text = unique_sentences[count]["original_text"],
                    clean_text = unique_sentences[count]["document_ids"],
                    embedding = unique_sentences[count]["embedding"]
                )

                session.execute(stmt)
                session.commit()

            else:
                sentence_index = seen_sentences[clean_sentence]
                unique_sentences[sentence_index]["document_ids"].append(unique_id)

            if count % BATCH_SIZE == 0:
                print(f"Loaded {count} sentences.")

            count += 1
            
    except Exception as e:
        print(f"Error processing document with id {unique_id}: {str(e)}")

faiss_index.add_with_ids(np.array(embeddings), np.array(sent_indexes))
faiss.write_index(faiss_index, "sentences.faiss")

def semantic_search(query):
    clean_query = process_text(query)
    query_embedding = embed([clean_query])
    
    D, I = faiss_index.search(query_embedding, k=10)
    
    results = []
    
    for i in range(I.shape[1]):
        index_id = int(I[0, i])
        similarity_score = D[0. i]
        
        stmt = select(sentences_table).where(sentences_table.c.sent_idx == index_id)
        result = session.execute(stmt).fetchone()
        
        if result is not None:
            document_ids = result[3]
            original_text = result[1]
            
            sentence_info = {
                "text": original_text,
                "document_ids": document_ids,
                "similarity_score": similarity_score
            }
            
            results.append(sentence_info)
        else:
            print(f"No result found for index_id {index_id}")
    
    return results

# Create FastAPI app
app = FastAPI(
    title="Semantic Search API", 
    description="This is a simple API for conducting semantic searches that utilizes Faiss and the Universal Sentence Encoder.", 
    version="1.0.0"
)

class Search(BaseModel):
    query: str

@app.get("/health")
def health_check():
    return {"message": "The server is running"}

@app.post("/search")
def search(search: Search):
    query = search.query
    results = semantic_search(query)
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
