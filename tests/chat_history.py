import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.2, max_tokens=50, google_api_key=GEMINI_API_KEY)



def process_user_query(user_query):
    prompt = create_prompt_template()
    formatted_prompt = prompt.format_prompt(user_query=user_query)
    response = model.invoke(formatted_prompt)
    
    return response.content


def create_prompt_template():
    template = '''
        You are an AI assistant.
        You should provide helpful and accurate responses to user queries.
        input: {user_query}
    '''


# while True:
#     user_query = input('\nAsk : ')
#     if user_query == 'quit': break

#     answer = process_user_query(user_query)
#     print(answer)




template = f"""
    You are an AI assistant.
    You should provide helpful and accurate responses to user queries.
"""

prompt = ChatPromptTemplate.from_messages(
    [("system", template,), MessagesPlaceholder(variable_name="messages"), ]
)
chain = prompt | model

template_content = prompt[0].prompt.template



store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(model, get_session_history)
model_with_memory = RunnableWithMessageHistory(chain, get_session_history)

config = {"configurable": {"session_id": "abc1"}}

response = with_message_history.invoke([HumanMessage(content=template_content)], config=config)


while True:
    query = input('\nAsk : ')
    if query == 'exit': break

    response = with_message_history.invoke([HumanMessage(content=query)], config=config)

    print('\nSystem : ', response.content)
    


# # print(store['abc1'])