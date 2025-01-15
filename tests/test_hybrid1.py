import os
import pandas as pd
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List
from rich import print

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return self.model.encode(documents, batch_size=32, show_progress_bar=True).tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode(query).tolist()

def create_documents_from_df(df):
    documents = []
    for index, row in df.iterrows():
        content = f"""
            Customer Type: {row['Customer Type']}
            Customer Name: {row['Customer Name']}
            Industry: {row['Industry']}
            Service Area: {row['Service Area']}
            Project Title: {row['Project/Case Study Title']}
            Description: {row['Short Description']}
            Keywords: {row['Key Words']}
            Tech Stack: {row['Tech Stacks']}
        """.strip()
        
        metadata = {
            'customer_type': row['Customer Type'],
            'customer_name': row['Customer Name'],
            'industry': row['Industry'],
            'service_area': row['Service Area'],
            'tech_stack': row['Tech Stacks']
        }
        
        documents.append(Document(page_content=content, metadata=metadata))
    
    return documents

def create_or_get_vector_storage(df, persist_directory, collection_name):
    documents = create_documents_from_df(df)
    
    keyword_retriever = BM25Retriever.from_documents(documents, preprocess_func=lambda text: text.lower(), k=15)

    if os.path.exists(persist_directory):
        vector_store = Chroma(persist_directory=persist_directory, collection_name=collection_name, embedding_function=embedding)
    else:
        vector_store = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=persist_directory, collection_name=collection_name)
    
    return vector_store, keyword_retriever

def create_prompt_template():
    text = """
        You are a specialized enterprise solution advisor analyzing case studies and project information. 
        You have access to a database of customer projects and case studies.

        Context information is formatted as follows for each case study:
        - Customer Type: ISV or Enterprise classification
        - Customer Name: Name of the customer
        - Industry: Customer's industry sector
        - Service Area: Type of service provided
        - Project Title: Title of the case study/project
        - Description: Detailed project description
        - Keywords: Relevant keywords
        - Tech Stack: Technologies used

        For the given query, analyze the provided context and:
        1. Consider ALL relevant case studies in the context
        2. If you find multiple relevant examples, include ALL of them
        3. If you can't find exact matches, include partially relevant examples
        4. Be explicit about why each case study is relevant to the query

        Context: {context}
        Query: {input}

        Important:
        - Never say information is unavailable without carefully checking the context
        - Include ALL relevant case studies, not just the most recent or best match
        - If information exists in the context, use it in your response
        - If you're unsure about details, quote the relevant context directly

        Please provide your response in this format for each relevant case study:

        Customer: [Name]
        Industry: [Industry]
        Relevance: [Why this case study is relevant to the query]
        Problem Statement: [Key challenges faced]
        Solution: [Implementation details]
        Technologies Used: [Tech stack]
        Impact: [Business outcomes]

    """
    return ChatPromptTemplate.from_messages([("system", text)])

def process_user_query(user_query, vectorDB, keywordDB):
    vectorstore_retriever = vectorDB.as_retriever(search_kwargs={"k": 15, "filter": None})
    
    ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retriever, keywordDB], weights=[0.7, 0.3])
    
    
    docs = ensemble_retriever.invoke(user_query)
    seen_content = set()
    unique_docs = []
    for doc in docs:
        if doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            unique_docs.append(doc)
    
    relevant_chunks = [doc.page_content for doc in unique_docs]
    print(relevant_chunks)
    context = "\n\n---\n\n".join(relevant_chunks)
    
    # prompt = create_prompt_template()
    # formatted_prompt = prompt.format_prompt(context=context, input=user_query)
    
    # response = llm.invoke(formatted_prompt)
    
    # return response.content
    return


load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding = SentenceTransformerEmbeddings(model_name)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=GEMINI_API_KEY, temperature=0.7)

def main(user_query, vectorDB, keywordDB):
    answer = process_user_query(user_query, vectorDB, keywordDB)
    return answer

if __name__ == "__main__":
    file_path = 'Nitor_Consolidated_List_of_Case_Studies_V2.csv'
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    df = df.astype(str).fillna('')
    
    persist_directory = './chroma_db_test1'
    collection_name = 'sales'
    vectorDB, keywordDB = create_or_get_vector_storage(df, persist_directory, collection_name)
    
    while True:
        que = input('\nAsk : ')
        if que == 'exit': break
        answer = main(que, vectorDB, keywordDB)
        print(answer)