from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

def load_embeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)

def load_llm(groq_api_key, model_name="Llama3-8b-8192"):
    return ChatGroq(groq_api_key=groq_api_key, model_name=model_name)

def load_prompt_template():
    return ChatPromptTemplate.from_template(
        """
        You are an expert in electrical and computer engineering, specializing in probability and random processes.
        Using only the information provided in the context below, answer the question as accurately and concisely as possible.
        If the answer is not present in the context, state that the information is not available.

        <context>
        {context}
        </context>

        Question: {input}
        """
    )
