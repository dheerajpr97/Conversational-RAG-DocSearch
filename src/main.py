import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.config import Config
from src.run_doc_qa_rag import run_doc_qa_rag
from src.run_conversational_rag import run_conversational_rag


## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=Config.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=Config.LANGCHAIN_PROJECT

# Load environment variables and set session state variables
st.session_state["groq_api_key"] = Config.GROQ_API_KEY
st.session_state["openai_api_key"] = Config.OPENAI_API_KEY


# Main function to run the Streamlit app
def main():

    # Set the background image
    background_image = """
    <style>
    [data-testid="stAppViewContainer"] > .main {
        background-image: url("https://archive.webdesignhot.com/wp-content/uploads/2013/04/Colorful-Abstract-Waves-on-Black-Background-Vector-Graphic_thumb.jpg");
        background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
        background-position: center;  
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>"""
    st.markdown(background_image, unsafe_allow_html=True)

    st.title("RAG with Multi-Functional Support")
    option = st.selectbox(
        "Choose a functionality",
        ("Select an option...", 
         "RAG Document Q&A With Multi-Lingual Support", "Conversational RAG with PDF Uploads and Interactive Q&A", ),
    )
    
    if option == "RAG Document Q&A With Multi-Lingual Support":
        run_doc_qa_rag()
    elif option == "Conversational RAG with PDF Uploads and Interactive Q&A":
        run_conversational_rag()
    elif option == "Select an option...":
        st.write("Please select a functionality to proceed.")


if __name__ == "__main__":
    main()