# Conversational-RAG-DocSearch

## Overview

**Conversational-RAG-DocSearch** is an advanced document search and question-answering system that leverages Retrieval-Augmented Generation (RAG) techniques with LangChain. It specializes in querying and exploring documents related to the fundamentals of probability and random processes, with support for multi-lingual document processing and interactive querying using state-of-the-art machine learning models.

The system provides robust features such as vector search using FAISS or Chroma and an intuitive, interactive UI powered by Streamlit. The latest version introduces the ability to conduct conversational queries with PDF uploads, maintain message history across sessions, and offers flexibility in choosing between different models and embedding options.

## Functionalities

Upon launching the Streamlit app, users are presented with two primary functionalities to choose from:

1. **RAG Document Q&A With Multi-Lingual Support for Probability and Random Processes**: 
   - Query in multiple languages using multilingual embeddings and advanced vector search techniques for answers on Probability and Random Processes for Electrical and Computer Engineers.
   
2. **Conversational RAG with PDF Uploads and Interactive Q&A**: 
   - Upload PDFs and engage in interactive, conversational queries, with the system maintaining message history for context.

## Features

- **Model and Embedding Flexibility**: Switch between using the Llama3 model from GROQ or OpenAI's models, and choose between Hugging Face's embeddings or OpenAI's embeddings to tailor the system to your specific requirements.
- **Multi-Lingual Support**: Process and query documents in multiple languages using multilingual embeddings.
- **Conversational RAG with PDF Uploads**: Upload PDFs and interactively query the content in a conversational manner.
- **Message History**: Maintain context across interactions, allowing for more coherent and contextually aware conversations.
- **Advanced Vector Search**: Choose between FAISS and Chroma for fast and efficient vector searches across documents.
- **Active Learning Feedback Loop**: Capture user feedback to continually improve the system's performance.
- **Interactive UI**: Enjoy an intuitive and user-friendly interface built with Streamlit for seamless interaction with your documents.

## Setup

### Prerequisites

- Python 3.8 or higher
- Install the required dependencies

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/Conversational-RAG-DocSearch.git
   cd Conversational-RAG-DocSearch
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**
   Create a `.env` file in the root directory and add the necessary API keys.
   ```bash
   GROQ_API_KEY=your_groq_api_key
   HF_TOKEN=your_huggingface_token
   OPENAI_API_KEY=your_openai_api_key
   ```

4. **Run the Streamlit App**
   ```bash
   streamlit run src/main.py
   ```

## Usage

- **Selecting a Functionality**: Upon launching the app, choose between "RAG Document Q&A With Multi-Lingual Support" and "Conversational RAG with PDF Uploads and Interactive Q&A".
- **Document Embedding**: Click the "Document Embedding" button to process and embed your documents.
- **Querying**: Enter a query to search the documents and receive answers based on the content.
- **Conversational Querying**: Upload a PDF and interact with the content using the conversational interface, with message history maintained across interactions.
- **Model and Embedding Selection**: Choose between GROQ Llama and OpenAI models, and between Hugging Face and OpenAI embeddings, directly within the UI.
- **Vector Database Selection**: Opt to use FAISS or Chroma for vector searches based on your performance or scalability needs.
- **Feedback**: Provide feedback on the accuracy of the answers to improve the system.

## Watch the Demo

[Demo Video](https://github.com/user-attachments/assets/a5d723a0-c1ef-411d-8971-3b9e7bafa946)

This video showcases the core features of the Conversational-RAG-DocSearch application. It walks through the user interface, highlighting how users can switch between functions, upload PDF documents, select different models and embeddings, and interact with the system through conversational queries. 

## Future Developments

- **Document Upload Enhancements**: Continue improving PDF processing capabilities with better error handling and user feedback integration.
- **Cloud Deployment**: Deploy the system on cloud platforms like AWS or GCP for better scalability.
- **Custom Model Fine-Tuning**: Fine-tune models on domain-specific corpora to enhance performance.
- **Real-Time Feedback Processing**: Implement real-time updates to the model based on user feedback.

## Acknowledgements

We would like to extend our gratitude to the following organizations and tools that made this project possible:

- **[GROQ](https://groq.com/)**: For providing the advanced LLM (Llama-3) used in this project.
- **[LangChain](https://langchain.com/)**: For offering a robust framework to build and integrate language model applications.
- **[Hugging Face](https://huggingface.co/)**: For providing multilingual embeddings and NLP models.
- **[OpenAI](https://openai.com/)**: For their state-of-the-art models and embedding options that enhance the system's versatility.
- **[Streamlit](https://streamlit.io/)**: For enabling the creation of an intuitive and interactive user interface.
- **[FAISS](https://github.com/facebookresearch/faiss)** and **[Chroma](https://www.trychroma.com/)**: For delivering efficient vector search capabilities that power the document retrieval process.

We appreciate the open-source community and the developers who contribute to these projects, making tools and resources freely available for everyone to use and build upon.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or new features.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
