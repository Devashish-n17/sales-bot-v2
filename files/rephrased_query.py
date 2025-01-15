import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=GEMINI_API_KEY)

def rephrased_query(user_query):
    # template = '''
    # #     You are an AI assistant that extracts relevant information from a structured dataset based on user queries. \
    # #     The dataset consists of past projects with details such as Client Name, Industry, Service Area, Tech Stacks, and Project Descriptions.

    # #     Task:
    # #     Understand the user query: Identify key components that match specific fields in the dataset.
    # #     Extract values based on field mapping:
    # #     Task:
    # #     Understand the user query: Identify key components that match specific fields in the dataset.
    # #     Identify Client Name, Industry, Service Area, Tech Stacks, or Project Descriptions from the user query.


    # #     input: {user_query}
    # #     Below is sample user query and sample response ;

    # #     User Query :  Have i worked on mobile application in fitness?
    # #     Output : Customer Type: enterprise | Industry: fitness | Service Area: mobile application development | Short Description: Searching for a mobile application project developed for the fitness ind
    # '''

    template = """You are an AI assistant specialized in analyzing project history data. Your role is to extract and match relevant information\
        from a structured database of past projects. Each project record contains the following fields:
        - Client Name
        - Industry
        - Service Area
        - Technology Stack
        - Project Description

        Your tasks:

        1. Analyze user queries to identify search criteria that correspond to database fields
        2. Extract and match relevant information based on the query context
        3. Present matching results in a structured format

        Field Matching Guidelines:
        - Map query keywords and context to appropriate database fields
        - Identify both explicit and implicit references to fields
        - Consider synonyms and related terms when matching

        Example:

        User Query: "Have I worked on mobile applications in fitness?
        Output : Industry: fitness | Service Area: mobile application development | Short Description: Searching for a mobile application project developed for the fitness industry.

        Note: Responses should be tailored to match only the relevant fields mentioned or implied in the user query. If possible please identify all the fields like client name, industry, service area, and enhance its short description according to the usecase.

        GOAL : MUST PROVIDE ALL FIELDS (INDUSTRy, CLIENT NAME, SERVICE AREA AND SHORT DESCRIPTION (30 WORDS))
    """

    messages = [("system", template, ),("human", user_query)]
    ai_msg = llm.invoke(messages)

    return ai_msg.content


if __name__ == '__main__':
    while True:
        user_query = input('\nAsk : ')
        if user_query == 'exit': break

        print(rephrased_query(user_query))


