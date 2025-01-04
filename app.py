import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd
import csv
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
import requests
from bs4 import BeautifulSoup

load_dotenv()

# Set environment variables
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="Llama3-8b-8192")

# Chat Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

def scrape_and_update():
    url = "https://genzmarketing.xyz/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    titles = [tag.text for tag in soup.find_all('h2')]
    contents = [p.text for p in soup.find_all('p')]
    urls = [a['href'] for a in soup.find_all('a', href=True)]

    # Ensure all lists have the same length
    min_length = min(len(titles), len(contents), len(urls))
    titles, contents, urls = titles[:min_length], contents[:min_length], urls[:min_length]

    data = pd.DataFrame({
        'title': titles,
        'url': urls,
        'content': contents
    })
    data.to_csv("genzmarketing_updated.csv", index=False)
    create_vector_embedding("genzmarketing_updated.csv")

# Preprocess CSV to handle errors
def preprocess_csv(file_path):
    try:
        df = pd.read_csv(file_path, quoting=csv.QUOTE_NONE, on_bad_lines='skip')
        cleaned_file_path = file_path.replace(".csv", "_cleaned.csv")
        df.to_csv(cleaned_file_path, index=False)
        return cleaned_file_path
    except Exception as e:
        st.error(f"Error preprocessing CSV: {e}")
        st.stop()

# Create Vector Embeddings
def create_vector_embedding(file_path):
    cleaned_file_path = preprocess_csv(file_path)
    st.session_state.loader = CSVLoader(file_path=cleaned_file_path)
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, embeddings)

# Streamlit UI
st.title("GenZ Contextual Chatbot: AI-Powered Solution for Accurate and Insightful Query Responses")
user_prompt = st.text_input("Enter your query from the GenZMarketing website.")

if st.button("Document Embedding"):
    create_vector_embedding("cleaned_genzmarketing_data.csv")

if st.button("Scrape and Update Data"):
    scrape_and_update()

if user_prompt and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': user_prompt})
    st.write(response['answer'])
    with st.expander("Document Similarity Search"):
        for doc in response['context']:
            st.write(doc.page_content)

# Schedule automatic scraping and updating
scheduler = BackgroundScheduler()
scheduler.add_job(scrape_and_update, 'interval', hours=24)
scheduler.start()
