import streamlit as st
import os
from pathlib import Path
from myfunctions import gen_audio, get_audio_query, no_chat_msg, update_chat_name, get_response

def bot_page():
    if st.session_state['current_chat']:
        st.title(f"{st.session_state['current_chat_name']}")
    else:
        st.title("RAGBot")

    # Ensure a chat is selected, if not prompt the user to start a new one
    if st.session_state['current_chat'] is None:
        no_chat_msg()

    # Initialize chat history for the selected session
    if st.session_state['current_chat'] not in st.session_state:
        st.session_state[st.session_state['current_chat']] = []

    # Create folder for the current chat session to save latest response audio
    current_chat_folder = f"Audio/{st.session_state['current_chat']}"
    Path(current_chat_folder).mkdir(parents=True, exist_ok=True)
        
    # Get audio query if available
    audio_query = get_audio_query()
    
    # Text input for user query
    text_query = None
    chat_input_msg = ""
    if st.session_state['lang']== 'en':
        chat_input_msg = "Your Message..."
    elif st.session_state['lang']=="fr":
        chat_input_msg = "Votre Message..."
    if st.session_state['current_chat'] is not None:
        text_query = st.chat_input(chat_input_msg)
    
    query = text_query if text_query else audio_query 

    if query:
        if len(st.session_state[st.session_state['current_chat']]) == 0:  # First query
            update_chat_name(query)
        response = get_response(query)
        

        # Convert response to speech and save it in the current chat folder
        audio_path = os.path.join(current_chat_folder, "response.mp3")
        gen_audio(response, audio_path)
        
    # Display chat messages for the current session
    if st.session_state[st.session_state['current_chat']]:
        for message in st.session_state[st.session_state['current_chat']]:
            if message["role"] == "user":
                st.chat_message("user").markdown(message["content"])
            else:
                st.chat_message("assistant").markdown(message["content"])

    # Play the bot's response audio from the current chat folder
    audio_file = os.path.join(current_chat_folder, "response.mp3")
    if os.path.exists(audio_file) and st.session_state[st.session_state['current_chat']]:
        st.audio(audio_file, autoplay = False)
    
