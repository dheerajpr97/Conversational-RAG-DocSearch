import streamlit as st
import time
from src.config import Config
from src.utils.docsearch_utils import create_vector_embedding
from src.query_processing import  load_prompt_template
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from src.utils.common_utils import get_model_and_embedding_choice,  handle_query
from src.utils.docsearch_utils import initialize_embeddings


# Main UI display function
def run_doc_qa_rag():
    # Set the title of the functionality
    st.markdown("<h1 style='font-size:32px;'>RAG Document Q&A With Multi-Lingual Support for Probability and Random Processes</h1>", unsafe_allow_html=True)

    # Get model and embedding choice from the user
    model_choice, embedding_choice = get_model_and_embedding_choice()

    # Initialize embeddings
    embeddings, use_cache = initialize_embeddings(model_choice, embedding_choice)
    if embeddings is None:
        return

    # Button to trigger document embedding creation
    if st.button("Document Embedding"):
        create_vector_embedding(embeddings, Config.PDF_DIRECTORY, use_cache=use_cache)
        st.success("Vector Database is ready")

    # Input field for user query
    user_prompt = st.text_area("Enter your query:", height=15)

    if user_prompt:
        # Load LLM based on model choice
        llm = handle_query(model_choice)
        
        # Load the prompt template
        prompt = load_prompt_template()
        
        # Create document chain and retrieval chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever(k=6)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Process the query and measure response time
        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        st.write(f"Response time: {time.process_time() - start:.2f} seconds")
        
        st.write(response['answer'])

        # Feedback mechanism
        feedback = st.radio("Is this answer helpful?", ("Yes", "No"))
        if feedback == "No":
            with open("feedback_log.txt", "a", encoding="utf-8") as f:
                f.write(f"User query: {user_prompt}\n")
                f.write(f"System response: {response['answer']}\n\n")

        # Expandable section for document similarity search results
        with st.expander("Document similarity Search"):
            for doc in response['context']:
                st.write(doc.page_content)
                st.write('------------------------')

if __name__ == "__main__":
    run_doc_qa_rag()