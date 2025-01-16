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
from db_conversations import store_conversation, get_last_conversations
from rich import print

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
file_path = 'Nitor_Consolidated_List_of_Case_Studies_V2.csv'
embedding = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004', google_api_key=GEMINI_API_KEY)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=GEMINI_API_KEY)

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


def process_user_query(user_query, vectorDB, keywordDB, chat_history):
    prompt = create_prompt_template()

    new_query = rephrased_query(user_query)
    vectorstore_retreiver = vectorDB.as_retriever(search_kwargs={"k": 10})
    ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retreiver, keywordDB], weights=[0.7, 0.3])
    
    docs = ensemble_retriever.invoke(new_query)
    relevant_chunks = [doc.page_content for doc in docs]
    formatted_prompt = prompt.format_prompt(context=relevant_chunks, user_query=user_query, chat_history=chat_history)
    response = llm.invoke(formatted_prompt)
    
    return response.content


def format_chat_history(past_conversations):
    chat_history = []
    for conv in past_conversations:
        chat_history.extend([HumanMessage(content=conv[0]), AIMessage(content=conv[1])])

    return chat_history


def create_prompt_template():
    template = """
        You are an expert Sales Solution Advisor, specifically designed to assist sales representatives during live client conversations.\
        Your primary role is to instantly retrieve and present relevant case studies, technology expertise, and project references in a clear, \
        persuasive format that can be easily communicated to clients.

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

        **Common Client Queries**
        - Technology expertise validation ("Have you worked with [technology]?")
        - Industry experience ("Show me examples in [industry]")
        - Similar solution requests ("Case studies for [solution type]")
        - Integration capabilities ("Experience with [platform] integration")
        - Implementation scale ("Projects similar to our size")

        **Response Structure**
        For Technology Expertise Queries:
        ```
        1. [Client Name] - [Industry]
            • Solution: [Brief description]
            • Impact: [Key metrics/outcomes]
        2. [Additional examples...]
        ```

        For Case Study Requests:
        ```
        1.  Client: [Company Name]
            • Business Challenge:  [1-2 lines about the problem]
            • Our Solution:  [2-3 key solution points]
            • Technologies Used:  [Core tech stack]
            • Business Impact:  [2-3 quantified results]
        2.  [Additional examples...]
        ```

        **Response Guidelines**
        1. CLARITY & BREVITY
        - Prioritize clear, scannable formats
        - Use bullet points for quick reference
        - Highlight quantifiable outcomes

        2. RELEVANCE & RECENCY
        - Prioritize cases from similar industries/scale
        - Focus on recent implementations
        - Highlight transferable successes

        3. CONFIDENCE BUILDING
        - Lead with strongest examples
        - Include success metrics when available
        - Mention relevant certifications/partnerships

        4. CUSTOMIZATION
        - Adapt detail level to query context
        - Include industry-specific terminology
        - Connect solutions to common pain points

        Chat History: {chat_history}
        Available Context: {context}
        Current Query: {user_query}

        Remember: Always provide a concise summary that a sales representative can easily communicate during a live call. Focus on business value and measurable outcomes.

        Response:
    """
    return ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_query}")
    ])


df = pd.read_csv(file_path, encoding='ISO-8859-1')
df = df.astype(str).fillna('')

persist_directory='./chroma_db'
collection_name='sales'
vectorDB, keywordDB = create_or_get_vector_storage(df, persist_directory, collection_name)


def main(user_query):
    past_conversations = get_last_conversations()
    print(past_conversations)
    chat_history = format_chat_history(past_conversations)

    answer = process_user_query(user_query, vectorDB, keywordDB, chat_history)
    store_conversation(user_query, answer)
    return answer


if __name__ == "__main__":
    while True:
        que = input('\nAsk : ')

        if que == 'exit': break

        answer = main(que.lower())
        print(answer)
        

