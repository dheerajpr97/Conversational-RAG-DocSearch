import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import streamlit as st
from src.config import Config
from src.ui import display_ui

# Load environment variables and set session state variables
st.session_state["openai_api_key"] = Config.OPENAI_API_KEY
st.session_state["groq_api_key"] = Config.GROQ_API_KEY

# Display the UI
if __name__ == "__main__":
    display_ui()
