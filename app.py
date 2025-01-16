import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd
import json
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
import subprocess
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from scrapy.http import HtmlResponse
import re
from gtts import gTTS
import tempfile

load_dotenv()

# Speech output function using gTTS for Streamlit compatibility
def speak_text(text):
    try:
        tts = gTTS(text=text, lang='en')
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file_path = temp_file.name + ".mp3"
        tts.save(temp_file_path)
        st.audio(temp_file_path, format="audio/mp3", start_time=0)
        os.remove(temp_file_path)  # Clean up after playing
    except Exception as e:
        st.error(f"Error with speech output: {e}")

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

# Web Scraper using Scrapy with Selenium
class MySpider(CrawlSpider):
    name = 'my_spider'
    allowed_domains = ['genzmarketing.xyz']
    start_urls = ['https://genzmarketing.xyz']
    custom_settings = {
        'DOWNLOAD_DELAY': 1,
        'ROBOTSTXT_OBEY': True,
        'CONCURRENT_REQUESTS': 10,
        'LOG_ENABLED': False,
        'FEED_FORMAT': 'json',
        'FEED_URI': 'genzmarketing_updated.json',
        'FEED_EXPORT_ENCODING': 'utf-8'
    }
    rules = (
        Rule(LinkExtractor(), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        options = Options()
        options.headless = True
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(response.url)
        html = driver.page_source
        driver.quit()
        selenium_response = HtmlResponse(url=response.url, body=html, encoding='utf-8')

        url = response.url
        title = selenium_response.css('title::text').get()
        content = selenium_response.css('p::text, div::text, span::text').getall()
        content = [text.strip() for text in content if text.strip()]

        if content:
            yield {
                'url': url,
                'title': title,
                'content': content
            }
        else:
            self.logger.warning(f"No content found on {url}")

def run_scraper():
    try:
        subprocess.run([
            "scrapy", "crawl", "my_spider"
        ], cwd="./genzmarketing_spider", check=True)
        if not os.path.exists('genzmarketing_updated.json'):
            st.error("Scraping failed or the file was not generated.")
    except subprocess.CalledProcessError as e:
        st.error(f"Error running the Scrapy spider: {e}")

# Data Cleaning Function
def clean_content(content):
    full_content = ' '.join(content)
    cleaned_content = re.sub(r'^Skip to:', '', full_content)
    cleaned_content = re.sub(r'\|\s*(\|\s*)+', ' ', cleaned_content)
    cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
    cleaned_content = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', cleaned_content)
    return cleaned_content.strip()

def create_vector_embedding(file_path):
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        st.stop()

    try:
        jq_schema = ".[]"
        st.session_state.loader = JSONLoader(file_path=file_path, jq_schema=jq_schema, text_content=False)
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, embeddings)
        st.success("Vector database created successfully!")
    except Exception as e:
        st.error(f"Error loading JSON file: {e}")

# Streamlit UI
st.title("GenZ Contextual Chatbot: AI-Powered Solution for Accurate and Insightful Query Responses")
user_prompt = st.text_input("Enter your query from the GenZMarketing website.")

if st.button("Scrape and Update Data"):
    run_scraper()
    if os.path.exists("genzmarketing_updated.json"):
        create_vector_embedding("genzmarketing_updated.json")
    else:
        st.error("Scraping failed or the file was not generated.")

if "scheduler" not in st.session_state:
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_scraper, 'interval', hours=24)
    scheduler.start()
    st.session_state["scheduler"] = scheduler

if user_prompt and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 5})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': user_prompt})
    st.write(response['answer'])
    speak_text(response['answer'])
    with st.expander("Document Similarity Search"):
        for doc in response['context']:
            st.write(doc.page_content)
