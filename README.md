# RAG-LangChain-DocSearch

## Overview

RAG-LangChain-DocSearch is a document search and question-answering system that utilizes Retrieval-Augmented Generation (RAG) techniques with LangChain specializing in fundamentals of probability and random processes. The system supports multi-lingual document processing and interactive querying using advanced machine learning models like HuggingFace's transformers and GROQ's LLMs. It provides features such as topic modeling, vector search using FAISS, and interactive visualization with Streamlit.

## Features

- **Multi-Lingual Support:** Processes and queries documents in multiple languages using multilingual embeddings.
- **Document Clustering and Topic Modeling:** Groups documents by topics using LDA or BERTopic and visualizes the distribution of topics.
- **Advanced Vector Search:** Uses FAISS for fast and efficient vector search across documents.
- **Active Learning Feedback Loop:** Captures user feedback to improve the system over time.
- **Interactive UI:** Provides an intuitive interface built with Streamlit for querying and exploring documents.


## Setup
### Prerequisites

Python 3.8 or higher
Install the required dependencies

### Installation
1. Clone the Repository 
```    
git clone https://github.com/yourusername/RAG-LangChain-DocSearch.git
cd RAG-LangChain-DocSearch
```
2. Install Dependencies

```
pip install -r requirements.txt
```

3. Set Up Environment Variables

Create a .env file in the root directory and add the necessary API keys.
```
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```
4. Run the Streamlit App

```
streamlit run src/main.py
```

## Usage
- Document Embedding: Click the "Document Embedding" button to process and embed your documents.
- Querying: Enter a query to search the documents and get answers based on the content.
- Feedback: Provide feedback on the accuracy of the answers to improve the system.

## Watch the Demo
 
https://github.com/user-attachments/assets/f6867223-270b-4e90-9681-c9a147069b95

This video demonstrates the functionality of the application. In this demo, you'll see how the application allows users to input complex queries specializing in fundamentals of probability and random processes, including mathematical expressions, and receive responses with properly formatted output. The application showcases the dynamic processing of user inputs, rendering of equations, and the overall user experience in a streamlined and intuitive interface.

## Future Enhancements
- Cloud Deployment: Deploy the system on cloud platforms like AWS or GCP for scalability.
- Custom Model Fine-Tuning: Fine-tune models on domain-specific corpora for improved performance.
- Real-Time Feedback Processing: Implement real-time updates to the model based on user feedback.





## Acknowledgements

We would like to extend our gratitude to the following organizations and tools that made this project possible:

- **[GROQ](https://groq.com/)**: For providing the advanced LLM (Llama-3) used in this project.
- **[LangChain](https://langchain.com/)**: For offering a robust framework to build and integrate language model applications.
- **[Hugging Face](https://huggingface.co/)**: For providing multilingual embeddings and NLP models.
- **[Streamlit](https://streamlit.io/)**: For enabling the creation of an intuitive and interactive user interface.
- **[FAISS](https://github.com/facebookresearch/faiss)**: For delivering efficient vector search capabilities that power the document retrieval process.
  <!-- - **[Gensim](https://radimrehurek.com/gensim/)**: For providing the topic modeling tools like LDA used in this project. 
  **[Transformers](https://github.com/huggingface/transformers)**: For offering foundational models and utilities that were integral to the project. -->
- **[OpenAI](https://openai.com/)**: For their contributions to the AI community and the APIs that enable advanced language processing.

We appreciate the open-source community and the developers who contribute to these projects, making tools and resources freely available for everyone to use and build upon.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or new features.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
