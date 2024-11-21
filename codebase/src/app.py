import streamlit as st
from src.disease_predictor import DiseasePredictor
from src.Symptoms_analyzer import SymptomAnalyzer
import time

st.set_page_config(
    page_title="MediMind - Medical Symptom Analyzer",
    page_icon="🏥",
    layout="wide"
)

st.info("""
    Tips for describing symptoms:
    - Be specific about what you're experiencing
    - Include duration of symptoms
    - Mention any relevant medical history
    """)

with st.sidebar:
    st.header("About")
    st.markdown("""
    MediMind combines advanced ML and LLM models to analyze your symptoms 
    and provide medical information.
    
    **Important:** This is not a substitute for professional medical advice.
    """)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state["chat_history"] = [{"role": "assistant", "content": "Hello! How can I help you today?"}]
if 'current_step' not in st.session_state:
    st.session_state["current_step"] = 'initial'
if 'symptoms' not in st.session_state:
    st.session_state["symptoms"] = ""
if 'follow_up_answers' not in st.session_state:
    st.session_state["follow_up_answers"] = {}

st.write(st.session_state)

@st.cache_resource
def load_models():
    try:
        disease_predictor = DiseasePredictor('models/pretrained_symtom_to_disease_model.pth')
        symptom_analyzer = SymptomAnalyzer()
        return disease_predictor, symptom_analyzer
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

disease_predictor, symptom_analyzer = load_models()
if not disease_predictor or not symptom_analyzer:
    st.error("Could not initialize the system. Please try again later.")

def streamer(text):
    for i in text:
        yield i
        time.sleep(0.001)


user_prompt = st.chat_input("Type your symptoms, you can also describe your medical history")

if user_prompt:
    st.session_state["chat_history"].append({"role": "user", "content": user_prompt})
    st.session_state["symptoms"] += user_prompt + " "
    
    if st.session_state["current_step"] == 'initial':
        disease_name = disease_predictor.predict_disease(user_prompt)
        st.session_state["chat_history"][0]["content"] += f" Based on the symptoms provided, the predicted disease is {disease_name}."
        st.session_state["current_step"] = 'follow_up'
        st.session_state["symptoms"] = ""
    
    if st.session_state["current_step"] == 'follow_up':
        follow_up_questions = symptom_analyzer.get_follow_up_questions(st.session_state["symptoms"])
        if follow_up_questions:
            for question in follow_up_questions:
                st.session_state["chat_history"].append({"role": "assistant", "content": question})
            st.session_state["current_step"] = 'follow_up_answers'
        else:
            st.session_state["current_step"] = 'end'

    if st.session_state["current_step"] == 'follow_up_answers':
        st.session_state["follow_up_answers"][st.session_state["symptoms"]] = user_prompt
        follow_up_questions = symptom_analyzer.get_follow_up_questions(st.session_state["symptoms"])
        if follow_up_questions:
            for question in follow_up_questions:
                st.session_state["chat_history"].append({"role": "assistant", "content": question})
        else:
            st.session_state["current_step"] = 'end'
    
    if st.session_state["current_step"] == 'end':
        st.session_state["chat_history"].append({"role": "assistant", "content": "Thank you for sharing your symptoms. I will now analyze them."})
        st.session_state["current_step"] = 'initial'
        st.session_state["symptoms"] = ""
        st.session_state["follow_up_answers"] = {}

st.markdown("""---""")