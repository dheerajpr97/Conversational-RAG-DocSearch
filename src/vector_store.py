import time
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import streamlit as st

def perform_topic_modeling(documents, num_topics=5):
    texts = [doc.page_content.split() for doc in documents]
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    
    for i, doc in enumerate(documents):
        doc_topic_dist = lda.get_document_topics(corpus[i])
        dominant_topic = max(doc_topic_dist, key=lambda x: x[1])[0]
        doc.metadata['topic'] = dominant_topic
    
    return documents

def create_vector_embedding(embeddings, pdf_directory):
    start_time = time.time()
    
    if "vectors" not in st.session_state:
        st.session_state.embeddings = embeddings
        st.session_state.loader = PyPDFDirectoryLoader(pdf_directory)
        
        st.write("Loading documents...")
        st.session_state.docs = st.session_state.loader.load()
        st.write(f"Loaded {len(st.session_state.docs)} documents in {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.write(f"Split into {len(st.session_state.final_documents)} document chunks in {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        st.session_state.final_documents = perform_topic_modeling(st.session_state.final_documents)
        st.write(f"Performed topic modeling in {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.write(f"Created vector embeddings in {time.time() - start_time:.2f} seconds")
