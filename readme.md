# GenZ Contextual Chatbot

## Project Overview
The GenZ Contextual Chatbot is an AI-powered solution designed to answer user queries by scraping data from the GenZMarketing website, processing it into vector embeddings, and retrieving the most accurate responses using a LangChain-powered LLM.

## Features
- **Data Scraping:** The system uses Scrapy with Selenium to scrape dynamic content from the GenZMarketing website.
- **Data Processing:** The scraped data is cleaned and stored as a JSON file for further processing.
- **Vector Embeddings:** The system uses FAISS and HuggingFace embeddings to create searchable vector databases.
- **Contextual Question Answering:** LangChain is used to retrieve the most relevant context and generate insightful responses.
- **Automated Updates:** The scraper runs periodically using APScheduler to ensure the data remains up-to-date.
- **Speech Output Integration:** Add Speech Output Integration in Chatbot.

## Technologies Used
- **Python** (Streamlit, Pandas, JSON)  
- **Scrapy** (for web scraping)  
- **Selenium** (for dynamic content extraction)  
- **LangChain** (for language model interaction)  
- **FAISS** (for vector database storage)  
- **HuggingFace Embeddings**  

## How It Works
1. **Scraping:** The Scrapy spider with Selenium extracts the entire website content, including dynamic data.  
2. **Processing:** The scraped data is stored in a JSON file and cleaned using a data cleaning function.  
3. **Embedding:** The cleaned data is converted into vector embeddings using FAISS and HuggingFace.  
4. **Query Handling:** User queries are compared against the vector database, and the most relevant documents are retrieved.  
5. **Answer Generation:** The LangChain-powered LLM generates detailed answers based on the retrieved documents.

## Project Setup
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/myself-nahid/GenZ-ChatBot.git
   cd GenZ-ChatBot
   ```
2. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set Up Environment Variables:**
   - Create a `.env` file with the following keys:
     ```plaintext
     GROQ_API_KEY=your_groq_api_key
     HF_TOKEN=your_huggingface_token
     ```
4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

## Usage
- **Scrape Website Data:** Click the `Scrape and Update Data` button.
- **Ask Questions:** Enter your query in the input field, and the chatbot will provide an insightful response based on the website content.

## Demos
- [result01](https://drive.google.com/file/d/16G0BqxRFjN3WZK_HRQ4ISx8pgOCaNMM_/view?usp=drive_link)
- [result02](https://drive.google.com/file/d/16DZqbMzLma06Z8g2cI4WqFLLBAkOvNAW/view?usp=drive_link)

## Troubleshooting
- **Scrapy Errors:** Ensure the working directory is correctly set and dependencies are installed.
- **Incomplete Answers:** Increase the document chunk size and overlap for better query handling.

## Future Improvements
- Add multilanguage support.
- Implement more advanced language models for deeper context understanding.
- Enable real-time website monitoring for data changes.

## License
This project is licensed under the MIT License.

