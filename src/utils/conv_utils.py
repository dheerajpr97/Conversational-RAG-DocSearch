import time
from langchain_community.vectorstores import FAISS, Chroma
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

def create_vector_embedding(embeddings, uploaded_files, vector_choice="FAISS"):
    
    documents = []
    for uploaded_file in uploaded_files:
        # Save the uploaded file temporarily
        temppdf = f"./temp_{uploaded_file.name}"
        with open(temppdf, "wb") as file:
            file.write(uploaded_file.getvalue())

        # Load and process the file
        loader = PyMuPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)

    # Splitting documents into chunks
    start_time = time.time()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    final_documents = text_splitter.split_documents(documents)
    st.write(f"Split into {len(final_documents)} document chunks in {time.time() - start_time:.2f} seconds")

    # Create embeddings for the documents
    start_time = time.time()
    if vector_choice == "FAISS":
        vectors = FAISS.from_documents(final_documents, embeddings)
    elif vector_choice == "Chroma":
        vectors = Chroma.from_documents(final_documents, embeddings)
    else:
        raise ValueError("Invalid vector choice")
    
    st.write(f"Created new temporary vector embeddings in {time.time() - start_time:.2f} seconds")    

    return vectors

