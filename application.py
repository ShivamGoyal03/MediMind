"""
MediMind - Medical Symptom Analyzer
This application uses a combination of a pre-trained RNN model and a large language model (LLM) to analyze user symptoms and provide medical information. The RNN model predicts potential diseases based on symptoms, while the LLM generates follow-up questions and medical recommendations.
The application is built using Streamlit and Hugging Face Transformers.

Usage:
    1. Run the app using the command: streamlit run application.py
    2. Enter your symptoms in the chat interface and follow the instructions to get predictions and recommendations.
    3. Click the "Reset Session" button to start a new session.

Project Structure:
    - data/: Contains the dataset Symptom2Disease.csv
    - models/: Contains the pre-trained RNN model (pretrained_symtom_to_disease_model.pth)
    - application.py: Streamlit app code for the MediMind application.
"""
import streamlit as st
import torch
from torch import nn
from transformers import pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
import warnings
warnings.filterwarnings('ignore')

# App Config
st.set_page_config(
    page_title="MediMind - Medical Symptom Analyzer",
    page_icon="🏥",
    layout="wide"
)
st.title("MediMind")

st.info("""
    Tips for describing symptoms:
    - Be specific about what you're experiencing
    - Include duration of symptoms
    - Mention any relevant medical history
    """)

# --- Disease Class Mapping ---
class_names = {
    0: 'Acne', 1: 'Arthritis', 2: 'Bronchial Asthma', 3: 'Cervical spondylosis',
    4: 'Chicken pox', 5: 'Common Cold', 6: 'Dengue', 7: 'Dimorphic Hemorrhoids',
    8: 'Fungal infection', 9: 'Hypertension', 10: 'Impetigo', 11: 'Jaundice',
    12: 'Malaria', 13: 'Migraine', 14: 'Pneumonia', 15: 'Psoriasis',
    16: 'Typhoid', 17: 'Varicose Veins', 18: 'allergy', 19: 'diabetes',
    20: 'drug reaction', 21: 'gastroesophageal reflux disease',
    22: 'peptic ulcer disease', 23: 'urinary tract infection'
}

with st.sidebar:
    st.header("About")
    st.markdown("""
    MediMind combines advanced ML and LLM models to analyze your symptoms 
    and provide medical information.  It uses a combination of a pre-trained RNN model and a large language model (LLM) for improved accuracy and context understanding.
    
    **Disclaimer:** This is not a substitute for professional medical advice.  Always consult a healthcare professional for diagnosis and treatment.
    """)
    doc_url = "https://medimind-doc.vercel.app/"
    st.sidebar.markdown(f'<a href="{doc_url}" target="_blank" style="text-decoration: none; color: white; background-color: #4CAF50; padding: 5px 10px; border-radius: 5px; display: inline-block; margin-bottom: 20px;">Go to Documentation</a>', unsafe_allow_html=True)


# --- NLTK and Data Loading ---
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

stemmer = SnowballStemmer(language='english')
english_stopwords = stopwords.words('english')


def load_data(filepath='data/raw/Symptom2Disease.csv'):
    """
    Load the symptom-disease dataset from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    try:
        df = pd.read_csv(filepath)
        df.drop(['Unnamed: 0'], axis=1, inplace=True, errors='ignore')
        df.drop_duplicates(inplace=True)
        train_data, _ = train_test_split(df, test_size=0.15, random_state=42)
        return train_data
    except FileNotFoundError:
        st.error(
            f"Error: Data file not found at '{filepath}'. Please provide the correct path.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        st.stop()

train_data = load_data()

# --- Tokenization and Vectorization ---
def tokenize(text):
    """
    Tokenize and stem the input text.
    
    Args:
        text (str): Input text to tokenize.
        
    Returns:
        List[str]: List of stemmed tokens
    """
    return [stemmer.stem(token) for token in word_tokenize(text)]


vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words=english_stopwords)
vectorizer.fit(train_data['text'])

# --- RNN Model ---
class RNN_model(nn.Module):
    """
    RNN model for symptom to disease prediction.

    Args:
        input_size (int): Input feature size.
        hidden_size (int): Hidden layer size.
        output_size (int): Output size (number of classes).

    Returns:
        torch.nn.Module: RNN model.

    Usage:
        model = RNN_model(input_size, hidden_size, output_size)
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=1, nonlinearity='relu', bias=True)
        self.output = nn.Linear(in_features=hidden_size,
                                out_features=output_size)

    def forward(self, x):
        y, hidden = self.rnn(x)
        x = self.output(y)
        return x

# Load the pre-trained RNN model
model_path = 'models/pretrained_symtom_to_disease_model.pth'
try:
    input_size = len(vectorizer.get_feature_names_out())
    output_size = len(class_names)
    rnn_model = RNN_model(input_size, 240, output_size)
    rnn_model.load_state_dict(torch.load(
        model_path, map_location=torch.device('cpu')))
    rnn_model.eval()
except FileNotFoundError:
    st.error(
        f"Error: RNN model file not found at '{model_path}'. Please provide the correct path.")
    st.stop()
except Exception as e:
    st.error(f"Error loading RNN model: {e}")
    st.stop()

# --- Disease Prediction (RNN) ---
def predict_disease(symptoms):
    """
    Predict the disease based on input symptoms using the RNN model.

    Args:
        symptoms (str): Input symptoms.

    Returns:
        str: Predicted disease.
    """
    try:
        transformed_symptoms = vectorizer.transform([symptoms])
        transformed_symptoms = torch.tensor(
            transformed_symptoms.toarray()).float()
        with torch.no_grad():
            output = rnn_model(transformed_symptoms)
            prediction = torch.argmax(torch.softmax(output, dim=1)).item()
        return class_names.get(prediction, "Unknown")
    except Exception as e:
        st.error(f"Error during RNN-based disease prediction: {e}")
        return "Unknown"


class LLMInterface:
    """
    Interface for the Large Language Model (LLM) for generating follow-up questions and recommendations.
    
    Args:
        model_name (str): meta-llama/Llama-3.2-1B-Instruct model from Hugging Face Transformers.

    Returns:
        LLMInterface: LLM interface object.

    Usage:
        >>> llm_int = LLMInterface()
        >>> questions = llm_int.get_follow_up_questions(initial_symptoms, conversation_history)
        >>> recommendations = llm_int.generate_recommendations(symptoms_info, diagnosis)

    More Info: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
    """
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct"):
        try:
            self.llm = pipeline(
                "text-generation",
                model=model_name,
                device_map="auto",
                truncation=True,
                pad_token_id=2
            )
        except Exception as e:
            st.error(
                f"Error loading LLM: {e}. Please ensure the model is accessible and your Hugging Face token is set.")
            st.stop()

    def get_follow_up_questions(self, initial_symptoms, conversation_history):
        """
        Generate follow-up questions based on initial symptoms and conversation history.

        Args:
            initial_symptoms (str): Initial symptoms provided by the user.
            conversation_history (str): Conversation history with the user.

        Returns:
            List[str]: List of follow-up questions.

        Usage:
            >>> questions = llm_int.get_follow_up_questions(initial_symptoms, conversation_history)
        """
        prompt = (
            f"You are a medical assistant. Based on the initial symptoms: '{initial_symptoms}', and the conversation history below, ask 3-4 DIFFERENT, concise, medically relevant follow-up questions to help determine the diagnosis. \n\nConversation History:\n{conversation_history}\n\nQuestions:"
        )
        try:
            response = self.llm(prompt, max_new_tokens=700, temperature=0.7,
                                return_full_text=False)[0]['generated_text']
            questions = [q.strip() for q in response.split('\n')
                         if q.strip() and '?' in q]
            return questions[:3]
        except Exception as e:
            st.error(f"Error generating follow-up questions: {e}")
            return ["How long have you had these symptoms?", "Is the pain constant or intermittent?", "What makes it better or worse?", "Are there any other symptoms?"]

    def generate_recommendations(self, symptoms_info, diagnosis):
        """
        Generate treatment and prevention recommendations based on symptoms and diagnosis.

        Args:
            symptoms_info (str): Information about the symptoms.
            diagnosis (str): Predicted disease diagnosis.

        Returns:
            str: Generated recommendations.

        Usage:
            >>> recommendations = llm_int.generate_recommendations(symptoms_info, diagnosis)
        """
        prompt = (
            f"Based on the symptoms and diagnosis below, provide concise treatment and prevention recommendations:\n\n"
            f"Symptoms: {symptoms_info}\n"
            f"Diagnosis: {diagnosis}\n\n"
            f"Recommendations:"
        )
        try:
            response = self.llm(prompt, max_new_tokens=700, temperature=0.5,
                                return_full_text=False)[0]['generated_text']
            return response.strip()
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
            return "Could not generate recommendations. Consult a doctor."


def chatbot_interface():
    """
    Streamlit chatbot interface for the MediMind application.
    
    Usage:
        - Run the app using the command: streamlit run application.py
        - Enter your symptoms in the chat interface and follow the instructions to get predictions and recommendations.
        - Click the "Reset Session" button to start a new session.
    """
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "disease_prediction" not in st.session_state:
        st.session_state.disease_prediction = None
    if "follow_up_questions" not in st.session_state:
        st.session_state.follow_up_questions = []
    if "follow_up_index" not in st.session_state:
        st.session_state.follow_up_index = 0
    if "follow_up_complete" not in st.session_state:
        st.session_state.follow_up_complete = False
    if "initial_prompt" not in st.session_state:
        st.session_state.initial_prompt = ""

    if st.button("Reset Session"):
        st.session_state.messages = []
        st.session_state.disease_prediction = None
        st.session_state.follow_up_questions = []
        st.session_state.follow_up_index = 0
        st.session_state.follow_up_complete = False
        st.session_state.initial_prompt = ""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not st.session_state.follow_up_complete:
        prompt = st.chat_input("Please describe your symptoms ....")
        if st.session_state.initial_prompt == "" and prompt:
            st.session_state.initial_prompt = prompt
            st.session_state.messages.append(
                {"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner('Processing...'):
                llm_int = LLMInterface()
                st.session_state.follow_up_questions = llm_int.get_follow_up_questions(
                    prompt, "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages]))

        # Ask follow-up questions one by one
        if st.session_state.follow_up_questions and st.session_state.follow_up_index < len(st.session_state.follow_up_questions):
            current_question = st.session_state.follow_up_questions[st.session_state.follow_up_index]
            user_answer = st.text_input(f"{current_question}")
            if user_answer:
                st.session_state.messages.append(
                    {"role": "user", "content": user_answer})
                st.session_state.follow_up_index += 1

    # Generate prediction and recommendations after all follow-up questions are answered
    if st.session_state.follow_up_questions and st.session_state.follow_up_index >= len(st.session_state.follow_up_questions):
        with st.spinner('Generating prediction and recommendations...'):
            llm_int = LLMInterface()
            st.session_state.disease_prediction = predict_disease(
                st.session_state.initial_prompt)
            all_info = "\n".join(
                [f"{m['role']}: {m['content']}" for m in st.session_state.messages])
            recommendations = llm_int.generate_recommendations(
                all_info, st.session_state.disease_prediction)
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Predicted Disease: {st.session_state.disease_prediction}\n\nRecommendations:\n{recommendations}"})
            with st.chat_message("assistant"):
                st.markdown(
                    f"Predicted Disease: {st.session_state.disease_prediction}\n\nRecommendations:\n{recommendations}")
        st.session_state.follow_up_complete = True

chatbot_interface()