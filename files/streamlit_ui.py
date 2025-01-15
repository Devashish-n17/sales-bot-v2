import streamlit as st
from query import main


st.title("Sales Chatbot")
st.write("Ask me anything!!!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("What's on your mind?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Generating Output..."):
            result = main(user_input)
            st.markdown(result)
    
    st.session_state.messages.append({"role": "assistant", "content": result})
