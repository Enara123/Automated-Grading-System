import pandas as pd
import javalang
from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np

# Load GCB model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
graphcodebert = RobertaModel.from_pretrained("microsoft/graphcodebert-base")
graphcodebert.eval()


def extract_features(code: str, tokenizer, graphcodebert, device, difficulty: str):
    """
    Extract combined features from code:
    - AST features using javalang
    - GraphCodeBERT embeddings
    - Metadata features (e.g., difficulty level)
    
    Returns a dictionary of feature_name -> value.
    """
    # 1. AST Features
    ast_features = extract_ast_features(code)

    # 2. GraphCodeBERT Embeddings
    gcb_features = get_code_embedding(code)

    # Metadata feature: difficulty level
    meta_feat = map_difficulty(difficulty)

    # 3. Combine all features into a single vector
    combined_features = combine_features(ast_features, gcb_features, meta_feat)

    return combined_features

# AST Feature Extraction
def extract_ast_features(code):
    """
    Extract basic AST features from Java code using javalang.
    Returns a dictionary of feature_name -> value.
    """
    features = {
        "num_classes": 0,
        "num_methods": 0,
        "num_variables": 0,
        "num_if": 0,
        "num_for": 0,
        "num_while": 0,
        "max_nesting_depth": 0
    }
    
    try:
        tree = javalang.parse.parse(code)
        
        for path, node in tree:
            node_type = type(node).__name__
            if node_type == "ClassDeclaration":
                features["num_classes"] += 1
            elif node_type == "MethodDeclaration":
                features["num_methods"] += 1
            elif node_type == "VariableDeclarator":
                features["num_variables"] += 1
            elif node_type == "IfStatement":
                features["num_if"] += 1
            elif node_type == "ForStatement":
                features["num_for"] += 1
            elif node_type == "WhileStatement":
                features["num_while"] += 1

        # Optional: compute max nesting depth
        def compute_depth(node, current_depth=0):
            children = list(node.children) if hasattr(node, "children") else []
            if not children:
                return current_depth
            return max([compute_depth(c, current_depth + 1) for c in children if isinstance(c, javalang.tree.Node)] + [current_depth])

        features["max_nesting_depth"] = compute_depth(tree)

    except javalang.parser.JavaSyntaxError:
        print("Failed to parse code; returning zeros")
        features = {k: 0 for k in features}

    return features

# GraphCodeBERT Embeddings Extraction
def get_code_embedding(code: str):
    """
    Returns a 768-dim embedding for the given code string using GraphCodeBERT [CLS] token.
    """
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = graphcodebert(**inputs)
    embedding = outputs.pooler_output  # shape: (1, 768)
    return embedding.squeeze(0).numpy()

#Difficulty Mapping
def map_difficulty(difficulty_str):
    """
    Map difficulty string to numerical value.
    Example mapping: 'Easy'->0, 'Medium'->1, 'Hard'->2
    """
    difficulty_map = {'Easy': 0, 'Medium': 1, 'Hard': 2}
    return difficulty_map.get(difficulty_str, 0)

# ATS + GCB Feature Combination
def combine_features(ast_features, gcb_features, meta_feat):
    """
    Combine AST features, GraphCodeBERT embeddings, and metadata feature into a single feature vector.
    ast_features: dict of feature_name -> value
    gcb_features: numpy array of shape (768,)
    meta_feat: float or int (e.g., difficulty level)
    Returns a combined numpy array.
    """

    ast_feat_arr = np.array(list(ast_features.values()), dtype=np.float32)
    meta_feat_arr = np.array([meta_feat], dtype=np.float32)
    
    combined = np.concatenate([ast_feat_arr, gcb_features, meta_feat_arr])

    return combined