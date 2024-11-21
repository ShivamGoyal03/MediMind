import torch
from torch import nn
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from src.utils.helpers import extract_symptoms

class RNN_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1080, hidden_size=240, num_layers=1, nonlinearity='relu', bias=True)
        self.output = nn.Linear(in_features=240, out_features=24)

    def forward(self, x):
        y, hidden = self.rnn(x)
        x = self.output(y)
        return x

class DiseasePredictor:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = RNN_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device),weights_only=True)
        self.model = self.model.to(self.device)  # Move model to device
        self.model.eval()
        self.all_symptoms = [
            'fever', 'cough', 'fatigue', 'difficulty breathing', 'headache',
            'body ache', 'sore throat', 'runny nose', 'nausea', 'vomiting',
            'diarrhea', 'chest pain', 'abdominal pain', 'joint pain', 'rash',
            'chills', 'sweating', 'loss of appetite', 'weight loss', 'dizziness',
            'muscle weakness', 'confusion', 'blurred vision', 'numbness', 'tingling',
            'swelling', 'constipation', 'anxiety', 'depression', 'insomnia',
            'memory problems', 'tremors', 'seizures', 'shortness of breath', 'wheezing',
            'itching', 'dry mouth', 'excessive thirst', 'frequent urination', 'hair loss',
            'bruising', 'bleeding', 'pale skin', 'jaundice', 'muscle cramps'
        ]
        
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([self.all_symptoms])
        
        self.disease_mapping = {
            0: "Common Cold",
            1: "Influenza",
            2: "Pneumonia",
            3: "Bronchitis",
            4: "Asthma",
            5: "COVID-19",
            6: "Sinusitis",
            7: "Gastroenteritis",
            8: "Food Poisoning",
            9: "Migraine",
            10: "Hypertension",
            11: "Diabetes",
            12: "Arthritis",
            13: "Allergies",
            14: "Urinary Tract Infection",
            15: "Thyroid Disorder",
            16: "Anemia",
            17: "Depression",
            18: "Anxiety",
            19: "Insomnia",
            20: "Eczema",
            21: "Psoriasis",
            22: "Celiac Disease",
            23: "Irritable Bowel Syndrome"
        }
    
    def preprocess_symptoms(self, symptoms_text):
        extracted_symptoms = extract_symptoms(symptoms_text)
        
        if not extracted_symptoms:
            raise ValueError("No recognized symptoms found in the input text.")
        
        binary_symptoms = self.mlb.transform([extracted_symptoms])
        
        padded_symptoms = np.pad(
            binary_symptoms, 
            ((0, 0), (0, 1080 - binary_symptoms.shape[1])),
            'constant'
        )
        
        symptoms_tensor = torch.FloatTensor(padded_symptoms).unsqueeze(1)
        symptoms_tensor = symptoms_tensor.to(self.device)  
        
        return symptoms_tensor
    
    def predict_disease(self, symptoms_text):
        try:
            if not symptoms_text or not isinstance(symptoms_text, str):
                raise ValueError("Please provide a valid symptoms description.")

            symptoms_tensor = self.preprocess_symptoms(symptoms_text)

            with torch.no_grad():
                output = self.model(symptoms_tensor)
                output = output.squeeze(0).squeeze(0)  
                probabilities = torch.softmax(output, dim=0)  
                predicted_idx = torch.argmax(probabilities).item()

            predicted_disease = self.disease_mapping.get(predicted_idx, "Unknown Disease")

            return {
                "disease": predicted_disease,
                "recognized_symptoms": extract_symptoms(symptoms_text)
            }

        except Exception as e:
            return f"Error in prediction: {str(e)}"