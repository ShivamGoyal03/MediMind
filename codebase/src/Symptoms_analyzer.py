import torch
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

class SymptomAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(f"Using device: {self.device}")
        
        self.llm = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.2-1B-Instruct",
            device_map="auto",
            truncation=True,
            pad_token_id=2  
        )

    def get_follow_up_questions(self, initial_symptoms):
        prompt = (
            f"Given the following initial symptoms described by a patient: {initial_symptoms}, "
            f"generate three of the most important follow-up questions a medical professional "
            f"should ask to gather more detailed and relevant information about the patient's condition. "
            f"Ensure that the questions are concise and medically relevant."
        )
        
        try:
            response = self.llm(
                prompt,
                max_length=1000,
                temperature=0.7,
                return_full_text=False
            )[0]['generated_text']
            
            # Extract questions
            questions = []
            for line in response.split('\n'):
                if '?' in line:
                    questions.append(line.strip())
            return questions[:3]  
        except Exception as e:
            return [
                "How long have you had these symptoms?",
                "Is the pain constant or intermittent?",
                "What makes it better or worse?"
            ]

    def predict_with_llm(self, symptoms_info):
        prompt = (
            f"You are a medical expert analyzing a patient's symptoms and history provided below:\n"
            f"Patient information: {symptoms_info}\n\n"
            f"Based on this information, analyze and provide the following:\n"
            f"1. The most likely medical condition(s) the patient might have.\n"
            f"2. Your confidence level in the assessment (High, Medium, Low).\n"
            f"3. A detailed reasoning for your assessment, linking symptoms to potential conditions.\n\n"
            f"Ensure the response is structured and easy to understand."
        )
        
        try:
            response = self.llm(
                prompt,
                max_length=2000,
                temperature=0.7,
                return_full_text=False
            )[0]['generated_text']
            
            # Ensure we have a valid response
            if not response.strip() or len(response) < 10:
                raise ValueError("No response generated by the model.")
                
            return response
            
        except Exception as e:
            # Fallback response
            return (
                "The system could not generate an analysis based on the provided symptoms. "
                "Here are general guidelines you may consider:\n"
                "1. Monitor the progression of your symptoms.\n"
                "2. Record any additional symptoms or changes over time.\n"
                "3. Consult a qualified healthcare professional for an accurate diagnosis and personalized treatment plan."
            )

    def get_recommendations(self, symptoms_info):
        prompt = (
            f"You are a healthcare professional tasked with providing recommendations based on the following symptoms "
            f"described by a patient: {symptoms_info}\n\n"
            f"Please provide detailed and structured guidance in the following format:\n"
            f"1. Immediate steps the patient can take to alleviate symptoms.\n"
            f"2. Lifestyle changes or preventive measures to avoid worsening the condition.\n"
            f"3. Clear guidance on when the patient should seek medical attention.\n\n"
            f"Ensure the recommendations are practical, evidence-based, and easy to follow."
        )
        
        try:
            response = self.llm(
                prompt,
                max_length=2000,
                temperature=0.7,
                return_full_text=False
            )[0]['generated_text']
            
            if not response.strip():  # If response is empty or invalid
                raise ValueError("No response generated by the model.")
            
            return response
            
        except Exception as e:
            return (
                "The system could not generate specific recommendations. "
                "Here are general guidelines to consider:\n"
                "1. Monitor your symptoms closely. If they worsen, seek medical attention immediately.\n"
                "2. Stay hydrated and rest as much as possible.\n"
                "3. Avoid any activities or substances that might aggravate your condition.\n"
                "4. Consult a healthcare professional for a detailed assessment."
            )

