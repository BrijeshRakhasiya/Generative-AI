import streamlit as st 
import os 
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS 
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
load_dotenv()
import time

# load the GROQ api key 

groq_api_key = os.getenv("GROQ_API_KEY")

st.title("Chat GROQ Demo ")

llm = ChatGroq(groq_api_key=groq_api_key , model="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only . 
    Please Provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question : {input}
    """
)

def vector_embedding():

    if "vectors" not in st.session_state : 

        st.session_state.embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.documents = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 200).split_documents(st.session_state.docs[:20])
        st.session_state.vectors =FAISS.from_documents(st.session_state.documents , st.session_state.embeddings )

prompt1 = st.text_input("Enter Your Question From Documents : ")

if st.button("Documents Embeddings ") :
    vector_embedding()
    st.write("Vector Store DB is Ready")


if prompt1: 
    document_chain = create_stuff_documents_chain(llm , prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever , document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input' : prompt1})
    print("Response time : " , time.process_time()-start)
    st.write(response['answer'])

    with st.expander("Document Similarity Search ") :

        for i , doc in enumerate(response['context']) : 
            st.write(doc.page_content)
            st.write("---------------------------------------------") 