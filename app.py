
import streamlit as st
import pickle
import pandas as pd

# Load Cut Label Encoder
with open("cut_le.pkl", "rb") as file:
    cut_le = pickle.load(file)

# Load Color Label Encoder
with open("color_le.pkl", "rb") as file:
    color_le = pickle.load(file)

# Load Clarity Label Encoder
with open("clarity_le.pkl", "rb") as file:
    clarity_le = pickle.load(file)

# Load Scaling model
with open("diamond_scaling.pkl", "rb") as file:
    scaler = pickle.load(file)

# Load KNN model
with open("diamond_knn.pkl", "rb") as file:
    model = pickle.load(file)

st.title("💎 Diamond Price Prediction")

st.write(
    "Enter the diamond details and click the Predict Price button.")

st.header("Enter Diamond Details")

# Carat
carat = st.number_input(

    "Carat",
    min_value=0.1,
    max_value=5.0,
    value=0.50
)

# Cut
cut = st.selectbox(
    "Cut",
    cut_le.classes_
)
# Color
color = st.selectbox(
    "Color",
    color_le.classes_
)
# Clarity
clarity = st.selectbox(
    "Clarity",
    clarity_le.classes_
)
# Depth
depth = st.number_input(
    "Depth",
    min_value=40.0,
    max_value=80.0,
    value=61.5
)
# Table
table = st.number_input(
    "Table",
    min_value=40.0,
    max_value=80.0,
    value=55.0
)
# X
x = st.number_input(
    "X",
    min_value=0.0,
    max_value=10.0,
    value=3.95
)
# Y
y = st.number_input(
    "Y",
    min_value=0.0,
    max_value=10.0,
    value=3.98
)
# Z
z = st.number_input(
    "Z",
    min_value=0.0,
    max_value=10.0,
    value=2.43
)

if st.button("🔮 Predict Price"):
    # Convert Cut into number
    cut_encoded = cut_le.transform([cut])[0]

    # Convert Color into number
    color_encoded = color_le.transform([color])[0]

    # Convert Clarity into number
    clarity_encoded = clarity_le.transform([clarity])[0]

    # These are the numerical columns
    numeric_data = pd.DataFrame(
        [[
            carat,
            depth,
            table,
            x,
            y,
            z
        ]],
        columns=[
            "carat",
            "depth",
            "table",
            "x",
            "y",
            "z"
        ]
    )
    scaled_data = scaler.transform(numeric_data)
    final_input = pd.DataFrame(
        [[
            cut_encoded,
            color_encoded,
            clarity_encoded,

            scaled_data[0][0],  # carat
            scaled_data[0][1],  # depth
            scaled_data[0][2],  # table
            scaled_data[0][3],  # x
            scaled_data[0][4],  # y
            scaled_data[0][5]   # z
        ]],
        columns=[
            "cut",
            "color",
            "clarity",
            "carat",
            "depth",
            "table",
            "x",
            "y",
            "z"
        ]
    )
    prediction = model.predict(final_input)
    st.success(prediction[0][0])