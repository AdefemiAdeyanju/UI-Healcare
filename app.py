import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PIL import Image

nltk.download('stopwords')
nltk.download('punkt')

# Load the trained model and vectorizer
with open('drug_prediction_model_nlp_sampled.pkl', 'rb') as model_file:
    classifier, vectorizer = pickle.load(model_file)

# Function to preprocess input text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_words)

def predict_drugs(diagnosis):
    # Preprocess input
    diagnosis = diagnosis.lower()
    diagnosis = preprocess_text(diagnosis)
    # Vectorize the input
    input_vectorized = vectorizer.transform([diagnosis])
    # Make prediction
    prediction = classifier.predict(input_vectorized)
    return prediction

# Streamlit UI


image = Image.open('drug.jpg')
st.image(image, caption='A Picture of Random Drugs')

st.title("Drug Prediction App")

# Input box for user to enter diagnosis
user_input = st.text_input("Enter Diagnosis:")

# Button to make prediction
if st.button("Predict"):
    # Make prediction
    result = predict_drugs(user_input)
    st.success(f"Predicted Drugs: {result}")

st.write('---')  
st.write('Please Note: The following app is not an alternative to drug prescriptions. Do meet an authorised medical personnel for that!')