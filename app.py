# Import standard libraries
import logging
import os
from concurrent.futures import ThreadPoolExecutor

# Import third-party libraries
import faiss
import nltk
import numpy as np
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import uvicorn
from fastapi import FastAPI
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from pydantic import BaseModel
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

stemmer = PorterStemmer()
stopwords_en = stopwords.words("english")

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.info("Starting script...")

# Utility functions
def load_data(data_path):
    with open(data_path, 'r') as f:
        data = f.readlines()
    return data

def preprocess_text(text):
    logging.info("Preprocessing text.")
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]
    words = [word for word in words if word not in stopwords_en]
    words = [stemmer.stem(word) for word in words]
    return " ".join(words)

def preprocess_data(data):
    with ThreadPoolExecutor() as executor:
        preprocessed_data = list(executor.map(preprocess_text, data))
    return preprocessed_data

# Load Universal Sentence Encoder
try:
    embed = hub.load("./universal-sentence-encoder")
    logging.info("Sentence encoder loaded from local 'models' directory.")
except Exception as e:
    logging.warning("Could not load the model locally. Error: {}".format(e))
    logging.warning("Attempting to download model from TensorFlow Hub...")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    logging.info("Sentence encoder downloaded from TensorFlow Hub.")

# Define index file paths
index_filepath_newsgroups = "newsgroups_index.faiss"
index_filepath_imdb = "imdb_index.faiss"
logging.info("Index file paths defined.")

# Initialize the indexes and documents dictionaries
indexes = {}
documents = {}

# Fetch newsgroups dataset and tokenize
if not os.path.exists(index_filepath_newsgroups):
    newsgroups_train = fetch_20newsgroups(subset="train")
    logging.info("Newsgroups data fetched and indexed.")

    documents_newsgroups = []
    sentences_newsgroups = []

    for i, document in tqdm(enumerate(newsgroups_train.data), total=len(newsgroups_train.data), desc="Processing newsgroups"):
        document = preprocess_text(document)
        document_sentences = sent_tokenize(document)
        sentences_newsgroups.extend(document_sentences)
        documents_newsgroups.extend([i] * len(document_sentences))

    # Vectorize sentences and create Faiss index
    sentence_vectors_newsgroups = embed(sentences_newsgroups).numpy()
    dimension_newsgroups = sentence_vectors_newsgroups.shape[1]
    index_newsgroups = faiss.IndexFlatL2(dimension_newsgroups)
    index_newsgroups.add(sentence_vectors_newsgroups)

    # Save the index to disk
    faiss.write_index(index_newsgroups, index_filepath_newsgroups)

else:
    index_newsgroups = faiss.read_index(index_filepath_newsgroups)
    newsgroups_train = fetch_20newsgroups(subset="train")

indexes["newsgroups"] = index_newsgroups
documents["newsgroups"] = newsgroups_train.data

# Fetch IMDB dataset and tokenize
if not os.path.exists(index_filepath_imdb):
    imdb_train = tfds.load("imdb_reviews", split="train[:5000]", shuffle_files=True)
    logging.info("IMDB data fetched and indexed.")

    documents_imdb = []
    sentences_imdb = []

    for i, example in tqdm(enumerate(imdb_train), total=5000, desc="Processing IMDB"):
        document = preprocess_text(example["text"].numpy().decode("utf-8"))
        document_sentences = sent_tokenize(document)
        sentences_imdb.extend(document_sentences)
        documents_imdb.extend([i] * len(document_sentences))

    # Vectorize sentences and create Faiss index
    sentence_vectors_imdb = embed(sentences_imdb).numpy()
    dimension_imdb = sentence_vectors_imdb.shape[1]
    index_imdb = faiss.IndexFlatL2(dimension_imdb)
    index_imdb.add(sentence_vectors_imdb)

    # Save the index to disk
    faiss.write_index(index_imdb, index_filepath_imdb)

else:
    index_imdb = faiss.read_index(index_filepath_imdb)
    imdb_train = tfds.load("imdb_reviews", split="train[:5000]", shuffle_files=True)
    sentences_imdb = [example["text"].numpy().decode("utf-8") for example in imdb_train]

indexes['imdb'] = index_imdb
documents['imdb'] = sentences_imdb

# Create FastAPI app
app = FastAPI(
    title="Semantic Search API", 
    description="This is a simple API for conducting semantic searches that utilizes Faiss and the Universal Sentence Encoder.", 
    version="1.0.0"
)
logging.info("FastAPI app created.")

class Index(BaseModel):
    name: str
    data_path: str

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
    # Load and preprocess the data
    data = load_data(index.data_path)
    preprocessed_data = preprocess_data(data)

    # Create and save the Faiss index
    index_vectors = embed(preprocessed_data).numpy()
    dimension = index_vectors.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(index_vectors)
    
    # Save the index to disk
    index_filepath = f"{index.name}_index.faiss"
    faiss.write_index(faiss_index, index_filepath)
    
    # Add to the in-memory index dictionary
    indexes[index.name] = faiss_index
    documents[index.name] = data
    
    return {"message": f"Index {index.name} created successfully"}

@app.get("/view_index/{index_name}")
async def view_index(index_name: str):
    if index_name in indexes:
        index = indexes[index_name]
        return {
            "Number of vectors": index.ntotal,
            "Vector dimension": index.d
        }
    else:
        return {"error": f"No index found with the name '{index_name}'."}

@app.post("/preprocess")
def preprocess_endpoint(text: str):
    return {"processed_text": preprocess_text(text)}

@app.post("/search")
def search_endpoint(search: Search):
    logging.info("Performing search.")
    query = search.query
    index_name = search.index
        
    if index_name not in indexes:
        return {"error": f"Invalid index name. Valid options are: {list(indexes.keys())}"}
        
    query = preprocess_text(query)
    query_vector = embed([query]).numpy()
    top_k = 10
    D, I = indexes[index_name].search(query_vector.astype(np.float32), top_k)
    results = [documents[index_name][i] for i in I[0]]
    return {"results": results}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
