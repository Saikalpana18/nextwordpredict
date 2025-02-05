import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the saved model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('next_word_model.h5')

# Load the saved tokenizer
@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as handle:
        return pickle.load(handle)

model = load_model()
tokenizer = load_tokenizer()

# Function to predict the next three words with probabilities
def predict_next_words(text, top_n=3):
    MAX_SEQUENCE_LEN = 5  # Ensure this matches training
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=MAX_SEQUENCE_LEN, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)[0]
    top_indices = np.argsort(predicted_probs)[-top_n:][::-1]
    top_words = [(tokenizer.index_word.get(idx, '(unknown)'), predicted_probs[idx]) for idx in top_indices]
    return top_words

# Streamlit UI
st.title("Next Word Predictor")
user_input = st.text_input("Enter a sentence:")

if st.button("Predict"):
    if user_input.strip():
        predictions = predict_next_words(user_input)
        for word, prob in predictions:
            st.write(f"Predicted word: **{word}** with probability: {prob:.4f}")
    else:
        st.write("Please enter some text.")