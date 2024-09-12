import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index=imdb.get_word_index()
rev_word_index={value:key for key,value in word_index.items()}

model=load_model('model.keras')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def decode_review(encoded_review):
    return ' '.join([rev_word_index.get(i-3,'?') for i in encoded_review])

def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)

    prediction=model.predict(preprocessed_input)

    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'

    return sentiment,prediction[0][0]
st.title('imdb sentiment analysis')

st.write('enter a movie review to classify it as positive or negative')

user_input=st.text_area('movie review')

if(st.button('classify')):
    preprocess_input=preprocess_text(user_input)

    prediction=model.predict(preprocess_input)

    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'

    st.write(f"sentiment: {sentiment}")
else:
    st.write('please enter  a movie review')