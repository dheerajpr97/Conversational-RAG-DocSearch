import time
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.utils.common_utils import load_vectors_from_disk, save_vectors_to_disk, load_embeddings, handle_openai_key


# Function to initialize embeddings based on user choices
def initialize_embeddings(model_choice, embedding_choice):
    if embedding_choice == "OpenAI" or model_choice.startswith("gpt"):
        openai_api_key = handle_openai_key()
        if not openai_api_key:
            return None, False
        
    if embedding_choice == "OpenAI" and model_choice.startswith("gpt"):
        st.info("You are using OpenAI embeddings with an OpenAI model.")
        return load_embeddings(embedding_choice, api_key=openai_api_key), False  # # Compute embeddings for OpenAI with GPT models
    
    elif embedding_choice == "Hugging Face" and model_choice.startswith("gpt"):
        st.info("You are using Hugging Face embeddings with an OpenAI model.")
        return load_embeddings(embedding_choice), True  # Use caching for Hugging Face embeddings with GPT models
    
    elif embedding_choice == "OpenAI" and not model_choice.startswith("gpt"):
        st.info("You are using OpenAI embeddings with a non-GPT model.")
        return load_embeddings(embedding_choice, api_key=openai_api_key), False # Compute embeddings for OpenAI with non-GPT models

    return load_embeddings(embedding_choice, api_key=None), True  # Use caching for Hugging Face with non-GPT models

CACHE_PATH = "vector_cache.pkl"

def create_vector_embedding(embeddings, pdf_directory,  use_cache=True):
    vectors = None
    
    if use_cache:
        vectors = load_vectors_from_disk(CACHE_PATH=CACHE_PATH)

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
            save_vectors_to_disk(vectors=st.session_state.vectors, CACHE_PATH=CACHE_PATH)

    return st.session_state.vectors