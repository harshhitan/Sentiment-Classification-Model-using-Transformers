import streamlit as st
from transformers import pipeline

st.title("Fine-Tuning BERT for IMDB Review Classification")

classifier = pipeline('text-classification', model='harshhitan/distilbert-base-uncased-sentiment-model')

text = st.text_area("Enter Your Review Here")
text = text[:512]

if st.button("Predict"):
        result = classifier(text)
        st.write("Prediction Result:", result)
