import torch
from src.preprocessing import preprocess_code
from src.feature_extraction import extract_features
from src.model import prediction
from transformers import RobertaTokenizer, RobertaModel

# Load model & tokenizer globally
tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = RobertaModel.from_pretrained("microsoft/graphcodebert-base")
model.eval()

def predict_grade(code: str, difficulty: str, trained_model, scaler, device):
    clean_code = preprocess_code(code)

    features = extract_features(clean_code, tokenizer, model, device, difficulty)

    grade = prediction(features, trained_model, scaler, device)

    return grade
