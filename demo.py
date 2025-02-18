import streamlit as st
from function_bot import groq_output_answer, gemini_output_answer
    
user_input = st.chat_input('Enter the message')

tab1, tab2 = st.tabs(["LLAMA", "GEMINI"])

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with tab1:
    if user_input:
        with st.chat_message("user"):
            #st.write("Hello 👋")
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.write(user_input)

        with st.chat_message("assistant"):
            answer = groq_output_answer(user_input)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.write(answer)

with tab2:
    if user_input:
        with st.chat_message("user"):
            #st.write("Hello 👋")
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.write(user_input)

        with st.chat_message("assistant"):
            answer = gemini_output_answer(user_input)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.write(answer)   