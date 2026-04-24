import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("lstm_model.h5")

st.title("LSTM Prediction")

input_data = st.text_input("Enter 3 values (comma separated)")

if st.button("Predict"):
    data = np.array([float(i) for i in input_data.split(",")])
    data = data.reshape(1, 3, 1)

    prediction = model.predict(data)
    st.write(prediction[0][0])