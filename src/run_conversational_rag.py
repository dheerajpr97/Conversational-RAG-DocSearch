import streamlit as st
import time

from src.query_processing import load_prompt_template, load_contextualize_q_prompt, load_final_qa_prompt
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from src.utils.common_utils import get_model_and_embedding_choice, handle_query, load_embeddings, handle_openai_key
from src.utils.conv_utils import create_vector_embedding

# Session history management function
def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

# Main  function
def run_conversational_rag():
    
    # Set the title of the functionality
    st.markdown("<h1 style='font-size:32px;'>Conversational RAG with PDF Uploads and Interactive Q&A</h1>", unsafe_allow_html=True)
    
    # Input for Session ID
    session_id = st.text_input("Session ID", value="default_session")

    # Initialize session state for chat history
    if 'store' not in st.session_state:
        st.session_state.store = {}

    # Track previous selections
    if 'prev_model_choice' not in st.session_state:
        st.session_state.prev_model_choice = None
    if 'prev_embedding_choice' not in st.session_state:
        st.session_state.prev_embedding_choice = None
    if 'prev_vector_choice' not in st.session_state:
        st.session_state.prev_vector_choice = None

    # Get model and embedding choice from the user
    model_choice, embedding_choice = get_model_and_embedding_choice()

    if model_choice.startswith("gpt") or embedding_choice == "OpenAI":
        openai_api_key = handle_openai_key()
        embeddings = load_embeddings(embedding_choice=embedding_choice, api_key=openai_api_key)
        if not openai_api_key:
            return None, False
    else:
        embeddings = load_embeddings(embedding_choice=embedding_choice, api_key=None)

    # Choose the vector database
    vector_choice = st.selectbox("Choose the vector database:", ("Select one..", "FAISS", "Chroma"))

    # Check if there was a change in model, embeddings, or vector choice
    if (model_choice != st.session_state.prev_model_choice or 
        embedding_choice != st.session_state.prev_embedding_choice or 
        vector_choice != st.session_state.prev_vector_choice):
        
        # Reset session state
        st.session_state.store = {}  # Clear chat history
        st.session_state.vectors = None  # Clear vector embeddings
        st.session_state.prev_model_choice = model_choice
        st.session_state.prev_embedding_choice = embedding_choice
        st.session_state.prev_vector_choice = vector_choice
        st.write("Model, embeddings, or vector store technique changed. Reinitializing...")

    # Store embeddings in session state
    st.session_state.embeddings = embeddings

    # File uploader for PDFs
    uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

    if uploaded_files and vector_choice != "Select one..":
        if 'vectors' not in st.session_state or st.session_state.vectors is None:
            st.write(f"Uploaded {len(uploaded_files)} files")
            st.session_state.vectors = create_vector_embedding(st.session_state.embeddings, 
                                                               uploaded_files, vector_choice)
            st.success("Temporary vector embeddings created for the session.")
        else:
            st.write("Using existing vector embeddings for this session.")

    # Handle user query
    user_prompt = st.text_area("Enter your query:", height=15)

    start = time.time()

    if user_prompt:
        # Load LLM based on model choice
        llm = handle_query(model_choice)

        # Contextualize the Question
        contextualize_prompt = load_contextualize_q_prompt()

        # Final QA Prompt
        final_qa_prompt = load_final_qa_prompt()

        # Retrieve Relevant Documents
        document_chain = create_stuff_documents_chain(llm, final_qa_prompt)
        retriever = st.session_state.vectors.as_retriever(k=6)
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)
        retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)
       
        conversation_rag_chain = RunnableWithMessageHistory(
            retrieval_chain, get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
        response = conversation_rag_chain.invoke(
            {"input": user_prompt},
            config={"configurable": {"session_id":session_id}
            }
        )
        st.write("User:", user_prompt)
        st.write("Assistant:", response['answer'])
        st.write(f"Time taken: {time.time() - start:.2f} seconds")

        # Feedback mechanism
        feedback = st.radio("Is this answer helpful?", ("Yes", "No"))
        if feedback == "No":
            with open("feedback_log.txt", "a", encoding="utf-8") as f:
                f.write(f"User query: {user_prompt}\n")
                f.write(f"System response: {response['answer']}\n\n")

if __name__ == "__main__":
    run_conversational_rag()
