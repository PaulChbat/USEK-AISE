import os
import streamlit as st
import openai
import os
from groq import Groq
import pyaudio
import wave
from deepgram import (DeepgramClient, SpeakOptions)
#------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to retrieve context from vectorstore based on the user's query
def retrieve_context(query: str, top_k: int = 3):
    # Using the vectorstore's retriever to fetch the most relevant documents
    vectorstore = st.session_state['vectorstore']
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query, k=top_k)
    # Combine the documents into a single context string
    context = "\n".join([doc.page_content for doc in docs])
    # Extract the sources from the metadata of each document
    sources = [doc.metadata.get("source", "Unknown source") for doc in docs]
    return context, sources

# Construct the prompt using the context
def create_prompt(context: str, query: str):
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    {context}

    Question: {question}
    Answer: """
    
    return template.format(context=context, question=query)

def get_response(query):
    # Combine all user queries in the chat history to create a contextual input
    all_queries = " ".join(
        message["content"] for message in st.session_state[st.session_state["current_chat"]] if message["role"] == "user"
    ) + " " + query
    
    # Retrieve context and sources
    context, sources = retrieve_context(all_queries)
    
    # Create the prompt using the retrieved context
    prompt = create_prompt(context, query)
    
    # Temporary history for the prompt and API call
    temp_history = st.session_state[st.session_state["current_chat"]] + [{"role": "user", "content": prompt}]
    
    # Call the local OpenAI model API
    openai.base_url = "http://127.0.0.1:1234/v1/"
    openai.api_key = "not-needed"
    completion = openai.chat.completions.create(
        model="local-model",
        messages=temp_history,
        temperature=0.1,
    )
    
    # Get the assistant's response
    response = completion.choices[0].message.content.strip()
    
    # Add sources to the response
    response += "\n\n ##### Sources:\n\n" + "\n\n".join([f"- {source}" for source in sources])
    
    # Save the query and response in the chat history (only the clean versions)
    st.session_state[st.session_state["current_chat"]].append({"role": "user", "content": query})
    st.session_state[st.session_state["current_chat"]].append({"role": "assistant", "content": response})
    
    return response


def create_chat_naming_prompt(input_text: str):
    template = """Summarize the given text into a short, descriptive title for a chat session, focusing on key terms. 
Retain any acronyms exactly as they are and do not include additional words like 'details' or 'information'. 
Do not answer the question if provided; just summarize it into a title.
Just return the summary:

Text:{input_text}

Summary:"""
    
    return template.format(input_text=input_text)

def get_chat_name(query):
    prompt = create_chat_naming_prompt(query)
    openai.base_url = "http://127.0.0.1:1234/v1/"
    openai.api_key = "not-needed"
    completion = openai.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.01,
    )
    name = completion.choices[0].message.content.strip()
    return name.replace('"', '')


#------------------------------------------------------------------------------------------------------------------------------------------------------
# Class used to format text loaded from website to be compatible with the chunking function of langchain
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def gen_audio(text, audio_path):
    sourceless_text = text.split("Sources")[0].strip() #remove sources from TTS 
    deepgram = DeepgramClient(os.environ.get("DEEPGRAM_API_KEY"))
    options = SpeakOptions(model='aura-asteria-en')
    deepgram.speak.v("1").save(audio_path, {"text":sourceless_text}, options)

def transcribe_audio(audio_file):
    client = Groq()
    with open(audio_file, "rb") as file:
        # Create a transcription of the audio file
        transcription = client.audio.transcriptions.create(
          file=(audio_file, file.read()), # Required audio file
          model="whisper-large-v3-turbo", # Required model to use for transcription
          prompt="Specify context or spelling",  # Change it to adapt to the context of the app
          response_format="json",  
          #language="en",  # Omitt to let Groq guess the language or you can specify one. 
          temperature=0.0  # The lower the temperature the more accurate the transcription is
        )
        return transcription.text

# Audio recording settings
FORMAT = pyaudio.paInt16  # Sets the audio format to a 16-bit signed integer
CHANNELS = 1  # Sets to mono channel recording, better for voice recognition
RATE = 44100  # Sample rate in Hz
CHUNK = 1024  # Chunk size, audio read in chunks of 1024 at a time
OUTPUT_FILE = "Audio/question.wav"  # Output file

# Initialize PyAudio
p = pyaudio.PyAudio()

# Start recording function
def start_recording():
    st.session_state['recording'] = True
    st.session_state['frames'] = []

    stream = p.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

    #with st.spinner("Recording audio..."):
    while st.session_state['recording']:
        data = stream.read(CHUNK)
        st.session_state['frames'].append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    

# Stop recording function
def stop_recording():
    st.session_state['recording'] = False
    
    # Save the recorded data as a WAV file
    wf = wave.open(OUTPUT_FILE, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(st.session_state['frames']))
    wf.close()

def get_audio_query():
    if os.path.exists("Audio/question.wav"):
        audio_query = transcribe_audio("Audio/question.wav")
        os.remove("Audio/question.wav") # Cleanup the audio file because i will have no use.
        return audio_query
    else:
        return None


def no_chat_msg():
    if st.session_state['lang']== 'en':
        st.markdown(
            """
            ### No chat selected
            Please select or create a new chat session from the sidebar.

            ### **Tips:**
            - Avoid changing the subject within the same chat. Instead, open a new chat for each new topic or question.
            """
        )
        return
    elif st.session_state['lang']=="fr":
        st.markdown(
            '''
            ### Aucun chat sélectionné
            Veuillez sélectionner ou créer une nouvelle session de chat dans la barre latérale.

            ### **Conseils :**
            - Évitez de changer de sujet dans le même chat. Ouvrez plutôt un nouveau chat pour chaque nouveau sujet ou question.
            '''
        )
        return
    
def update_chat_name(query):
    """Update the current chat name based on the provided query, ensuring no name conflicts."""
    new_chat_name = get_chat_name(query)

    # Check for existing names to avoid conflicts
    existing_names = {chat[1] for chat in st.session_state['chat_sessions']}
    
    # If the new name already exists, modify it
    if new_chat_name in existing_names:
        # Find a unique name by appending a number
        suffix = 1
        while f"{new_chat_name} ({suffix})" in existing_names:
            suffix += 1
        new_chat_name = f"{new_chat_name} ({suffix})"
    
    # Update the chat session
    for chat in st.session_state['chat_sessions']:
        if chat[0] == st.session_state['current_chat']:
            chat[1] = new_chat_name
            st.session_state['current_chat_name'] = new_chat_name
            break





