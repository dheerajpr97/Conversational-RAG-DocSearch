# vector_store.py

import os
import pickle
import time
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

CACHE_PATH = "vector_cache.pkl"

def save_vectors_to_disk(vectors):
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(vectors, f)

def load_vectors_from_disk():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'rb') as f:
            vectors = pickle.load(f)
        return vectors
    return None

def create_vector_embedding(embeddings, pdf_directory, use_cache=True):
    vectors = None
    
    if use_cache:
        vectors = load_vectors_from_disk()

    if vectors is not None:
        st.session_state.vectors = vectors
        st.success("Loaded vector embeddings from disk.")
    else:
        # Proceed with document processing if vectors not found or caching is not applicable
        start_time = time.time()
        st.session_state.embeddings = embeddings
        st.session_state.loader = PyPDFDirectoryLoader(pdf_directory)
        
        st.write("Loading documents...")
        st.session_state.docs = st.session_state.loader.load()
        st.write(f"Loaded {len(st.session_state.docs)} documents in {time.time() - start_time:.2f} seconds")

        # Splitting documents
        start_time = time.time()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.write(f"Split into {len(st.session_state.final_documents)} document chunks in {time.time() - start_time:.2f} seconds")

        # Create vector embeddings
        start_time = time.time()
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.write(f"Created vector embeddings in {time.time() - start_time:.2f} seconds")

        if use_cache:
            save_vectors_to_disk(st.session_state.vectors)

    return st.session_state.vectors
