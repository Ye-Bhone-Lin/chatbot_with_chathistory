import pickle
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

def groq_output_answer(user_input: str) -> str:
    
    load_dotenv()
    groq_api_key = os.environ.get('GROQ_API_KEY')

    with open("rag_components.pkl", "rb") as f:
        data = pickle.load(f)

    # Restore components
    retriever = data["retriever"]
    contextualize_q_prompt = data["contextualize_q_prompt"]
    qa_prompt = data["qa_prompt"]

    model =  ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", groq_api_key=groq_api_key)

    # Recreate history-aware retriever

    history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(model,qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)
    chat_history = []
    user_input = user_input
    message1 = rag_chain.invoke({"input": user_input,"chat_history":chat_history})
    return message1['answer']

def gemini_output_answer(user_input: str) -> str:
    
    load_dotenv()
    gemini_api_key = os.environ.get('GEMINI_API_KEY')

    with open("rag_components.pkl", "rb") as f:
        data = pickle.load(f)

    # Restore components
    retriever = data["retriever"]
    contextualize_q_prompt = data["contextualize_q_prompt"]
    qa_prompt = data["qa_prompt"]

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp",google_api_key=gemini_api_key,temperature=0.2,max_tokens=None)

    # Recreate history-aware retriever

    history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(model,qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)
    chat_history = []
    user_input = user_input
    message1 = rag_chain.invoke({"input": user_input,"chat_history":chat_history})
    return message1['answer']