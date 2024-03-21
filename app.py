import io
import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize Streamlit session states
if 'vectorDB' not in st.session_state:
    st.session_state.vectorDB = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'bot_name' not in st.session_state:
    st.session_state.bot_name = ""

# Function to extract text from a PDF file
def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text: str):
    # This function will split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_response(query):
    """This function will return the output of the user query!"""
    chain = load_qa_chain(OpenAI(model_name="gpt-3.5-turbo-16k", api_key=openai_api_key), chain_type="stuff")
    response = chain.run(question=query, input_documents=st.session_state.vectorDB.similarity_search(query=query))
    return response

def get_vectorstore(text_chunks):
    """This function will create a vector database and store the embedding of the text chunks into the VectorDB"""
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

if __name__ == '__main__':
    st.set_page_config(page_title="Chatbot", page_icon="🤖")
    st.title('🤖 RAG Chatbot 🧠')

    # User inputs
    pdf_content = st.file_uploader("Upload PDF Content:", type='pdf')

    # Process PDF and create vector database
    if pdf_content:
        with st.spinner('Processing PDF...'):
            vectorDB = get_vectorstore(get_text_chunks(get_pdf_text(pdf_content)))
            st.session_state['vectorDB'] = vectorDB
            st.success('PDF processed successfully!', icon="✅")

    # If the vector database is ready, show the chatbot interface
    if st.session_state.vectorDB:
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Taking the input i.e. query from the user
        if prompt := st.chat_input("Ask a Question"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.write(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                response = get_response(prompt)
                st.write(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})