from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Prompt template for base question answering
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

# Additional prompt for contextualizing questions in a conversational setting
def load_contextualize_q_prompt():
    return ChatPromptTemplate.from_messages(
        [
            ("system", 
            "Based on the provided chat history and the latest user question, "
            "identify if the question references previous context. If it does, "
            "rephrase the question so that it becomes fully self-contained and understandable "
            "on its own, without any need for prior context."
            "Your task is solely to reformulate the question if necessary, and otherwise return it as is."
            ),

            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

# Additional prompt for final question answering
def load_final_qa_prompt():
    return ChatPromptTemplate.from_messages(
        [
            ("system", 
            "You are a knowledgeable assistant tasked with answering questions based on provided context. "
            "Use the information within the given context to generate an accurate and concise response. "
            "If the necessary information to answer the question is not present in the context, clearly state that the "
            "information is unavailable. Do not add any information not found in the context. Format and display equations" 
            "and derivations as an output in LaTeX with matrices, fractions, subscripts, superscripts, Greek symbols, integrals, "
            "and modulus functions even while solving problems stick to the same to display equations."
            "\n\n"
            """<context>
            {context}
            </context>"""
            ),

            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
