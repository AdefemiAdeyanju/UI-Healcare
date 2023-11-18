from fastapi import FastAPI, Form

import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = FastAPI()

#nltk.download('stopwords')
#nltk.download('punkt')

# Load the trained model and vectorizer
classifier, vectorizer = joblib.load('drug_prediction_model_nlp_sampled.joblib')

# Function to preprocess input text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_words)

@app.post("/predict")
def predict_diagnosis(diagnosis: str = Form(...)):
    # Preprocess input
    diagnosis = diagnosis.lower()
    diagnosis = preprocess_text(diagnosis)

    # Vectorize the input
    input_vectorized = vectorizer.transform([diagnosis])

    # Make prediction
    prediction = classifier.predict(input_vectorized)
    
    return {"diagnosis": diagnosis, "predicted_drugs": prediction[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
