
import time
import streamlit as st
from llm_model import VRAGLLMModel
from ww_model import WakeWordClassifier
from asr_model import AutomaticSpeechRecognition
from tts_model import TextToSpeechModel

#Set the application title
st.title(":blue[_Voice_] RAG :sound:")

st.sidebar.markdown("* Loading Llama 3.2 model")

#Initialize the large language model
if "vrag_model" not in st.session_state:
    vrag_model = VRAGLLMModel()
    vrag_model.load_model()
    st.session_state["vrag_model"] = vrag_model

st.sidebar.markdown("* Llama loaded")

st.sidebar.markdown("* Loading ASR model")
#Initialize the automatic speech recognition model
if "asr_model" not in st.session_state:
    asr_model = AutomaticSpeechRecognition()
    asr_model.load_asr_model()
    st.session_state["asr_model"] = asr_model

st.sidebar.markdown("* ASR model is loaded")

st.sidebar.markdown("* Loading TTS model")
#Initialize the text to speech model
if "tts_model" not in st.session_state:
    tts_model = TextToSpeechModel()
    tts_model.load_tts_model()
    st.session_state["tts_model"] = tts_model

st.sidebar.markdown("* TTS model is loaded")

# Streamed response emulator
def response_generator(query):
    llm = st.session_state["vrag_model"]
    response = llm.generate_response(query)
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input(" "):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

st.sidebar.markdown("*  Loading wake word model")


def detect_wakeword():
    ww_status = st.session_state.ww_model.detect_wakeword()
    return ww_status

def transcribe_speech():
    prompt = st.session_state.asr_model.convert_speech_to_text(out=True)
    return prompt

def get_rag_response(prompt):
     # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    return response

def text_to_speech(response):
    st.session_state.tts_model.convert_text_to_speech(response)
    return 'Done'

#Main function to conduct voice based rag
def run_rag():
    st.session_state['running_vrag'] = True
    ww_status = False
    while ww_status == False:
        ww_status = detect_wakeword()
    if(ww_status):
        print('Wake word detected')
        prompt = transcribe_speech()
        # Add user message to the chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat  container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        response = get_rag_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        text_to_speech(response)
    st.session_state['running_vrag'] = False
    time.sleep(1)
    return

#Initialize the wake word classifier
if "ww_model" not in st.session_state:
    ww_model = WakeWordClassifier()
    ww_model.load_wakeword_classifier()
    st.session_state["ww_model"] = ww_model
    st.sidebar.markdown("* Wake word detection model loaded")
    

while True:
    if "running_vrag" not in st.session_state or st.session_state.running_vrag == False:
        run_rag()



        

