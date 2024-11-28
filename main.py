import streamlit as st
from chatbot import bot_page
from myfunctions import start_recording, stop_recording
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# Set up the Streamlit page
st.set_page_config(page_title="RAGBot", layout="wide")

# Default Page
if 'page' not in st.session_state:
    st.session_state['page'] = "Chatbot"

# Initialize variables in session_state
if 'chat_sessions' not in st.session_state:
    st.session_state['chat_sessions'] = []  # List of all chat sessions
if 'current_chat' not in st.session_state:
    st.session_state['current_chat'] = None  # Track current active chat
if 'chat_counter' not in st.session_state:
    st.session_state['chat_counter'] = 1   # Counter for how many chats were created
if 'current_chat_name' not in st.session_state:
    st.session_state['current_chat_name'] = None # Generated name for the current chat
if 'lang' not in st.session_state: 
    st.session_state['lang'] = 'en' # Current session language

if 'vectorstore' not in st.session_state:
    used_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name = used_model_name)
    try:
        vectorstore = FAISS.load_local('db_faiss_usek_1', embeddings, allow_dangerous_deserialization=True)
        st.session_state['vectorstore'] = vectorstore
    except FileNotFoundError:
        st.error("No saved USEK DB found!")

# Sidebar 
with st.sidebar:
    st.write("## Navigation")
    
    if st.button("New Chat", help="Create a new chat session"):
        # Create a new chat session
        new_chat_id = st.session_state['chat_counter']
        new_chat_name = f"Chat {new_chat_id}" # initially set the name to 'Chat id'
        st.session_state['chat_sessions'].append([new_chat_name,new_chat_name])
        st.session_state['current_chat_name'] = new_chat_name
        st.session_state['current_chat'] = new_chat_name
        st.session_state['page'] = "Chatbot"  # Ensure the page switches to the new chat
        st.session_state['chat_counter'] += 1  # Increment the counter for the next chat
    
    if st.session_state['current_chat']:
        # Audio Input section
        st.write("## Record your question")
        col1, col2 = st.columns([0.5, 0.5])
        col1.button('▶', on_click=start_recording, help="Start Recording")
        col2.button('🔴', on_click=stop_recording, help="Stop Recording")
    
    # Display buttons for each chat session
    if st.session_state['chat_sessions']:
        st.write("## Chats")
    for chat_id, chat_name in st.session_state['chat_sessions']:
        col1, col2 = st.columns([0.7, 0.3])  # Create two columns
        with col1:
            if st.button(chat_name):
                st.session_state['current_chat_name'] = chat_name
                st.session_state['current_chat'] = chat_id
                st.session_state['page'] = "Chatbot"  # Switch to chatbot page when a chat is selected
        with col2:
            if st.button("🗑️", key=f"delete_{chat_id}", help="Delete"):  # Unique key for each delete button
                st.session_state['chat_sessions'].remove([chat_id, chat_name])
                # Check if the deleted chat was the active one
                if st.session_state['current_chat'] == chat_id:
                    # Set current chat to None or the first available chat
                    if st.session_state['chat_sessions']:
                        st.session_state['current_chat'] = st.session_state['chat_sessions'][0][0]  # Set to first available chat
                        st.session_state['current_chat_name'] = st.session_state['chat_sessions'][0][1]
                    else:
                        st.session_state['current_chat'] = None  # No chats available

                st.rerun()  # Refresh the page to reflect changes

# Navigation based on selected page
if st.session_state['page'] == "Chatbot":
    bot_page()