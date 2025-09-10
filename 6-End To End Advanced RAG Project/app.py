import streamlit as st 
import os
import time
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if "vector" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.documemts = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 200).split_documents(st.session_state.docs)
    st.session_state.db = FAISS.from_documents(st.session_state.documemts , st.session_state.embeddings)


st.title("Chat Groq Demo ")

llm = ChatGroq(groq_api_key=groq_api_key , model="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the Questions based upon on hte provided contect only . 
    Please provide the most accurate response based on the question .
    <context>
    {context}
    <context>
    Question: {input}
    """
)

document_chain = create_stuff_documents_chain(llm , prompt)

retriever = st.session_state.db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever , document_chain)

prompt = st.text_input("Input your prompt Here ")

if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input" : prompt})
    print("Response Time : " , time.process_time() - start)
    st.write(response['answer'])

    with st.expander("Document Similarity Search") : 
        for i , doc in enumerate(response['answer']) : 
            st.write(doc.page_content)
            st.write("----------------------------------------")
