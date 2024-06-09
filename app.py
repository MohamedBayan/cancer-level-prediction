import streamlit as st
from keras.models import load_model
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# Load the saved model
NN = False

# Create a dropdown list with single selection
selected_option = st.selectbox(
    'Select the model',
    [
        'decision_tree_model',
        'gradient_boosting_model',
        'nearest_neighbors_model',
        'Neural Network',
        'multilayer_perceptron_model',
        'naive_bayes_model',
        'random_forest_model',
        'svm'
    ],
    help="Choose the model you want to use for prediction."
)
st.write('You selected:', selected_option)

# Load the selected model
if selected_option.endswith('Network'):
    loaded_model = tf.keras.models.load_model("model.h5")
    NN = True
else:
    loaded_model = joblib.load(selected_option)

# Title of the app
st.title('Health Risk Prediction')

# Dictionary to hold feature names and their corresponding ranges
features = {
    'Air Pollution': (1, 8),
    'Alcohol use': (1, 8),
    'Dust Allergy': (1, 8),
    'Occupational Hazards': (1, 8),
    'Genetic Risk': (1, 7),
    'Chronic Lung Disease': (1, 7),
    'Balanced Diet': (1, 8),
    'Obesity': (1, 7),
    'Smoking': (1, 8),
    'Passive Smoker': (1, 8),
    'Chest Pain': (1, 9),
    'Coughing of Blood': (1, 9),
    'Fatigue': (1, 9),
    'Weight Loss': (1, 8),
    'Shortness of Breath': (1, 9),
    'Wheezing': (1, 8),
    'Swallowing Difficulty': (1, 8),
    'Clubbing of Finger Nails': (1, 9),
    'Frequent Cold': (1, 7),
    'Dry Cough': (1, 7),
    'Snoring': (1, 7)
}

# Create sliders for each feature
selected_values = {}
for feature, (min_val, max_val) in features.items():
    selected_values[feature] = st.slider(
        feature,
        min_val,
        max_val,
        min_val,
        help=f"Select the value for {feature}"
    )

# Show the entered values
sample_input = np.array([[float(value) if value != '' else np.nan for value in selected_values.values()]])

if st.button('Make Prediction'):
    if NN:
        prediction = loaded_model(sample_input)
        prediction = np.argmax(prediction)
    else:
        prediction = loaded_model.predict(sample_input)

    label = ""
    if prediction == 0:
        label = "Low"
    elif prediction == 1:
        label = "Medium"
    else:
        label = "High"

    st.write('Prediction:', label)
