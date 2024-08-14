import streamlit as st
import time
from src.vector_store import create_vector_embedding
from src.query_processing import load_embeddings, load_llm, load_prompt_template
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import plotly.express as px

def plot_topic_distribution(documents):
    topics = [doc.metadata['topic'] for doc in documents if 'topic' in doc.metadata]
    fig = px.histogram(topics, title="Topic Distribution Across Documents", labels={'value': 'Topic', 'count': 'Number of Documents'})
    st.plotly_chart(fig)

def display_ui():
    st.title("RAG Document Q&A With Multi-Lingual Support with GROQ And Llama3")

    if st.button("Document Embedding"):
        embeddings = load_embeddings()
        create_vector_embedding(embeddings, "src/books")
        st.success("Vector Database is ready")
        plot_topic_distribution(st.session_state.final_documents)  # Add this line to visualize topics

    user_prompt = st.text_input(
        "Enter your query for the fundamentals on PROBABILITY AND RANDOM PROCESSES FOR ELECTRICAL AND COMPUTER ENGINEERS"
    )

    if user_prompt:
        groq_api_key = st.session_state.get("groq_api_key", "")
        llm = load_llm(groq_api_key)
        prompt = load_prompt_template()
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        st.write(f"Response time: {time.process_time() - start:.2f} seconds")

        st.write(response['answer'])

        feedback = st.radio("Is this answer helpful?", ("Yes", "No"))
        if feedback == "No":
            with open("feedback_log.txt", "a", encoding="utf-8") as f:
                f.write(f"User query: {user_prompt}\n")
                f.write(f"System response: {response['answer']}\n\n")

        with st.expander("Document similarity Search"):
            for doc in response['context']:
                st.write(doc.page_content)
                st.write('------------------------')
