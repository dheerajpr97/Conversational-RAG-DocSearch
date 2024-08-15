import re
import streamlit as st
import plotly.express as px
from src.query_processing import load_embeddings, load_llm


# # Function to detect LaTeX equations with subscripts, superscripts, Greek letters, integrals, and modulus functions
# def contains_latex(text):
#     latex_patterns = [
#         r"\\frac",    # Fraction
#         r"\\lim",     # Limit
#         r"\\int",     # Integral
#         r"\\int_{[^}]*}^{[^}]*}",  # Integral with limits
#         r"\\sum",     # Summation
#         r"\\sum_{[^}]*}^{[^}]*}",  # Summation with limits
#         r"\\sum",     # Summation
#         r"\\sqrt",    # Square root
#         r"\\left", r"\\right",  # For modulus, norms, and other paired delimiters
#         r"\\|",       # Modulus |...|
#         r"\\\|",      # Norm or modulus (e.g., ||x||)
#         r"_[{]?[a-zA-Z0-9]+[}]?",  # Subscript (matches '_x', '_{xy}', etc.)
#         r"\^[{]?[a-zA-Z0-9]+[}]?",  # Superscript (matches '^2', '^{xy}', etc.)
#         r"\\phi", r"\\tau", r"\\alpha", r"\\beta", r"\\gamma", r"\\delta",  # Greek letters
#         r"\\tilde",   # Tilde notation (e.g., \tilde{X})
#         r"~",         # Tilde character used for non-breaking space or in math
#         r"\\begin", r"\\end",  # Begin and end environments
#         r"\$.*\$",    # Inline math mode $
#         r"\\\(", r"\\\)",  # LaTeX inline math mode \( \)
#         r"\\\[", r"\\\]",  # LaTeX display math mode \[ \]
#     ]
#     return any(re.search(pattern, text) for pattern in latex_patterns)


# Function to plot topic distribution across documents
def plot_topic_distribution(documents):
    topics = [doc.metadata['topic'] for doc in documents if 'topic' in doc.metadata]
    fig = px.histogram(topics, title="Topic Distribution Across Documents", labels={'value': 'Topic', 'count': 'Number of Documents'})
    st.plotly_chart(fig)

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

# Function to initialize embeddings based on user choices
def initialize_embeddings(model_choice, embedding_choice):
    if embedding_choice == "OpenAI" or model_choice.startswith("gpt"):
        openai_api_key = handle_openai_key()
        if not openai_api_key:
            return None, False
        
    if embedding_choice == "OpenAI" and model_choice.startswith("gpt"):
        st.info("You are using OpenAI embeddings with an OpenAI model.")
        return load_embeddings(embedding_choice, openai_api_key), False  # Enable caching for OpenAI embeddings with GPT models
    
    elif embedding_choice == "Hugging Face" and model_choice.startswith("gpt"):
        st.info("You are using Hugging Face embeddings with an OpenAI model.")
        return load_embeddings(embedding_choice), True  # Enable caching for Hugging Face embeddings with GPT models

    return load_embeddings(embedding_choice), True  # Use caching for Hugging Face with non-GPT models

# Function to handle query processing based on selected model
def handle_query(model_choice):
    if model_choice == "Llama3":
        groq_api_key = st.session_state.get("groq_api_key", "")
        return load_llm(groq_api_key, model_name="Llama3-8b-8192")
    else:
        openai_api_key = st.session_state.get("openai_api_key", "")
        return load_llm(openai_api_key, model_name=model_choice)

# Function to display the response with LaTeX formatting
def display_response(response_text):
    # Split response into parts that are text and those that are equations
    if response_text.startswith("$$") and response_text.endswith("$$"):
        # Remove the $$ at the beginning and end
        equation_text = response_text.strip("$$")
        st.latex(equation_text)
    else:
        # Detect and format inline LaTeX within text
        inline_latex_parts = re.split(r'(\$\$[^\$]*\$\$)', response_text)  # Split by double $$
        for part in inline_latex_parts:
            if part.startswith("$$") and part.endswith("$$"):
                # It's LaTeX content
                st.latex(part.strip("$$"))
            else:
                # It's plain text
                st.write(part)