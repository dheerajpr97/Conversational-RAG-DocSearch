import os
import pickle
import re
import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings


def load_vectors_from_disk(CACHE_PATH):
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'rb') as f:
            vectors = pickle.load(f)
        return vectors
    return None

def save_vectors_to_disk(CACHE_PATH, vectors):
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(vectors, f)

# Function to get model and embedding choices from the user
def get_model_and_embedding_choice():
    model_choice = st.selectbox("Choose the model:", ("Llama3", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"))
    embedding_choice = st.selectbox("Choose the embeddings model:", ("Hugging Face", "OpenAI"))
    return model_choice, embedding_choice

# Function to handle OpenAI API key input
def handle_openai_key():
    openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key to proceed.")
        return None
    st.session_state["openai_api_key"] = openai_api_key
    return openai_api_key

# Function to load embeddings based on user choices
def load_embeddings(embedding_choice, api_key=None):
    if embedding_choice == "Hugging Face":
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2") 
    elif embedding_choice == "OpenAI" and api_key:
        return OpenAIEmbeddings(openai_api_key=api_key)
    else:
        raise ValueError("Invalid embedding choice or missing API key for OpenAI")

# Function to load LLM based on user choices   
def load_llm(api_key, model_name="Llama3-8b-8192"):
    if "Llama3" in model_name:
        return ChatGroq(groq_api_key=api_key, model_name=model_name)
    else:
        return ChatOpenAI(openai_api_key=api_key, model_name=model_name)
    
# Function to handle query processing based on selected model
def handle_query(model_choice):
    if model_choice == "Llama3":
        groq_api_key = st.session_state.get("groq_api_key", "")
        return load_llm(groq_api_key, model_name="Llama3-8b-8192")
    else:
        openai_api_key = st.session_state.get("openai_api_key", "")
        return load_llm(openai_api_key, model_name=model_choice)