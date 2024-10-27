# MediMind

**Problem Statement:**
The AI-powered Medical Q&A Chatbot aims to assist users by interpreting their inputted symptoms using Natural Language Processing (NLP) and providing relevant responses, including possible diagnoses, preventive measures, and treatment suggestions. The chatbot serves as an informational tool, offering preliminary insights while encouraging users to seek professional medical consultation. This project focuses on creating a user-friendly, privacy-conscious system that facilitates medical Q&A interactions without storing user data. Additionally, the chatbot utilizes Hugging Face base models and datasets, fine-tuned specifically for medical diagnosis to enhance the accuracy and relevance of its diagnostic capabilities.

---

**Solution:**
The AI-powered Medical Q&A Chatbot, MediMind, is a web-based application that leverages the Hugging Face Transformers library to provide accurate and helpful responses to medical questions. The chatbot is built using Streamlit, a Python library for creating web applications, and is designed to be user-friendly and easy to deploy. MediMind utilizes a large dataset of medical questions and answers to provide accurate responses and can be trained on new data to improve its accuracy and knowledge. The chatbot is intended to assist users by interpreting their inputted symptoms and providing relevant responses, including possible diagnoses, preventive measures, and treatment suggestions. It serves as an informational tool, offering preliminary insights while encouraging users to seek professional medical consultation. The chatbot is designed to be privacy-conscious and does not store user data, ensuring the confidentiality and security of user information.

## Features

- User-friendly interface
- Accurate and helpful responses to medical questions
- Ability to train on new data to improve accuracy
- No storage of user data
- Encourages users to seek professional medical consultation

## Getting Started

1. Clone the repository
2. Install the dependencies
3. Run the app

```bash
git clone https://github.com/ShivamGoyal03/MediMind.git
cd MediMind
pip install -r requirements.txt
streamlit run app.py
```