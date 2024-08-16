from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

def load_embeddings(embedding_choice, api_key=None):
    if embedding_choice == "Hugging Face":
        return HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    elif embedding_choice == "OpenAI" and api_key:
        return OpenAIEmbeddings(openai_api_key=api_key)
    else:
        raise ValueError("Invalid embedding choice or missing API key for OpenAI")
    

def load_llm(api_key, model_name="Llama3-8b-8192"):
    if "Llama3" in model_name:
        # Load GROQ's Llama3 model
        return ChatGroq(groq_api_key=api_key, model_name=model_name)
    else:
        # Load OpenAI's GPT models
        return ChatOpenAI(openai_api_key=api_key, model_name=model_name)

def load_prompt_template():
    return ChatPromptTemplate.from_template(
        """
        You are an expert in electrical and computer engineering, specializing in probability and random processes.
        Using only the information provided in the context below, answer the question as accurately and concisely as possible.
        Format and display equations and derivations as an output in LaTeX with fractions, subscripts, superscripts, Greek symbols, integrals, and modulus functions
        even while solving problems stick to the same to display equations. 
        If the answer is not present in the context, state that the information is not available.

        <context>
        {context}
        </context>

        Question: {input}
        """
    )
