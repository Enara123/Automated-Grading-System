import streamlit as st
from src.predict import predict_grade
from src.model import GradingNN  
import joblib
import torch

# Load model & scaler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_model = torch.load("model/grading_model2.pth", map_location=device, weights_only=False)
trained_model.eval()
trained_model.to(device)
scaler = joblib.load("model/scaler.pkl")


st.set_page_config(page_title="Code Grading System", layout="wide")

st.title("Automated Code Grading System")

uploaded_file = st.file_uploader("Upload your code file (.java)", type=["java"])

difficulty = st.selectbox("Select Difficulty Level", ["Easy", "Medium", "Hard"])

if uploaded_file is not None:
    code = uploaded_file.read().decode("utf-8")
    st.text_area("Preview Code", code, height=200)

    if st.button("Grade Code"):
        with st.spinner("Evaluating..."):
            grade = predict_grade(code, difficulty, trained_model, scaler, device)
        st.success(f"✅ Predicted Grade: {grade}")
