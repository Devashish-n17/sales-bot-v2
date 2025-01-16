import os
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.schema import Document, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from rephrased_query import rephrased_query
from rich import print

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
file_path = 'Nitor_Consolidated_List_of_Case_Studies_V2.csv'
embedding = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004', google_api_key=GEMINI_API_KEY)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=GEMINI_API_KEY)

prompt = None
def initialize_metadata_and_prompt():
    global prompt
    if prompt is None: prompt = create_prompt_template()
    return prompt


def create_or_get_vector_storage(df, persist_directory, collection_name):
    chunks = []
    for index, row in df.iterrows():
        chunk = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        chunks.append(chunk)

    documents = [Document(page_content=chunk, metadata={}) for chunk in chunks]

    keyword_retriever = BM25Retriever.from_documents(documents)

    if os.path.exists(persist_directory):
        vector_store = Chroma(persist_directory=persist_directory, collection_name=collection_name, embedding_function=embedding)
    else:
        vector_store = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=persist_directory, collection_name=collection_name)
    return vector_store, keyword_retriever


def process_user_query(user_query, vectorDB, keywordDB):
    global prompt
    if prompt is None: prompt = initialize_metadata_and_prompt()
    new_query = rephrased_query(user_query)
    vectorstore_retreiver = vectorDB.as_retriever(search_kwargs={"k": 10})
    ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retreiver, keywordDB], weights=[0.5, 0.5])
    
    docs = ensemble_retriever.invoke(new_query)
    relevant_chunks = [doc.page_content for doc in docs]
    formatted_prompt = prompt.format_prompt(context=relevant_chunks, rephrased_query=new_query, user_query=user_query)
    response = llm.invoke(formatted_prompt)
    
    return response.content


def create_prompt_template():
    template = """
        You are a specialized enterprise solution advisor with expertise in analyzing and retrieving relevant case studies and project information. \
        You have access to a comprehensive database of customer projects and case studies across various industries, technologies, and service areas.

        **Data Schema Understanding** 
        The data contains detailed project information with the following key fields:
        - Customer Type (ISV/Enterprise)
        - Customer Name
        - Industry
        - Service Area
        - Project/Case Study Title
        - Short Description
        - Key Words
        - Tech Stacks

        **Example queries**
        1. "Have I worked on Data Warehousing?"
        2. "Show case studies related to mobile app development in healthcare."
        3. "What are the key technologies used in CRM modernization projects?"
        4. "Provide examples of deep learning applications in automotive."
        5. "Which enterprise clients have used SAP-based solutions?"

        
        **Primary Functions:**
        1. **CASE STUDY SUMMARIZATION**
        - Extract key insights from project descriptions.
        - Provide a structured summary with problem, solution, and impact.
        - Ensure clarity and brevity for quick reference.

        2. **TECHNOLOGY & INDUSTRY MAPPING**
        - Identify core technologies used in similar projects.
        - Highlight relevant industry expertise.
        - Suggest similar case studies where applicable.

        **Response Format:**
        - Provide a structured summary:
        ```
        <Customer Name> - <Industry>
        - Challenge: <Brief problem statement>
        - Solution: <Concise solution implemented>
        - Business Impact: <Key benefits and outcomes>
        ```
        - If multiple case studies are relevant, list them sequentially.

        **Response Guidelines:**
        - Focus on clarity, relevance, and brevity.
        - Prioritize recent and impactful projects.
        - Ensure responses are easy to understand and actionable.
        
        **Query Processing:**
        1. Parse the query to extract industry, technology, and solution needs.
        2. Retrieve relevant case studies from stored chunks.
        3. Generate a concise, professional summary for sales team reference.

        context: {context}
        input: {user_query}

        answer:

    """
    return ChatPromptTemplate.from_messages([("system", template), MessagesPlaceholder(variable_name="chat_history"), ("human", "{user_query}")])


df = pd.read_csv(file_path, encoding='ISO-8859-1')
df = df.astype(str).fillna('')

persist_directory='./chroma_db_test1'
collection_name='sales'
vectorDB, keywordDB = create_or_get_vector_storage(df, persist_directory, collection_name)


def main(user_query):
    answer = process_user_query(user_query, vectorDB, keywordDB)
    return answer


if __name__ == "__main__":
    while True:
        que = input('\nAsk : ')

        if que == 'exit': break

        answer = main(que.lower())
        print(answer)
        

