#!/usr/bin/env python3
"""
Debug trained model embedding generation
"""

import sys
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINED_MODEL_PATH = os.path.join(BASE_DIR, "trained_toolbench_retriever")
BEST_MODEL_PATH = os.path.join(BASE_DIR, "trained_toolbench_retriever_best")
TOOLBENCH_MODEL_NAME = "ToolBench/ToolBench_IR_bert_based_uncased"

def debug_embeddings():
    """Debug embedding generation for both models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test queries
    test_queries = [
        "I need to check the health of the SQUAKE API and get project information",
        "I want to track a package through Correo Argentino",
        "I'm looking for cocktail recipes and financial data"
    ]
    
    # Load ToolBench model
    print("\n📦 Loading ToolBench model...")
    toolbench_model = SentenceTransformer(TOOLBENCH_MODEL_NAME)
    toolbench_model = toolbench_model.to(device)
    toolbench_model.eval()
    print(f"✅ ToolBench model loaded")
    
    # Load trained model
    print("\n📦 Loading trained model...")
    if os.path.exists(BEST_MODEL_PATH):
        trained_model = SentenceTransformer(BEST_MODEL_PATH)
        print(f"✅ Trained model loaded from {BEST_MODEL_PATH}")
    elif os.path.exists(TRAINED_MODEL_PATH):
        trained_model = SentenceTransformer(TRAINED_MODEL_PATH)
        print(f"✅ Trained model loaded from {TRAINED_MODEL_PATH}")
    else:
        print(f"❌ No trained model found")
        return
    
    trained_model = trained_model.to(device)
    trained_model.eval()
    
    # Test embedding generation
    print("\n🧪 Testing embedding generation...")
    
    for i, query in enumerate(test_queries):
        print(f"\n--- Query {i+1}: '{query}' ---")
        
        # ToolBench model
        with torch.no_grad():
            toolbench_emb = toolbench_model.encode([query], convert_to_tensor=True, device=device)
            toolbench_emb_norm = F.normalize(toolbench_emb, p=2, dim=-1)
            toolbench_emb_np = toolbench_emb_norm.cpu().numpy()
        
        print(f"ToolBench embedding shape: {toolbench_emb_np.shape}")
        print(f"ToolBench embedding norm: {np.linalg.norm(toolbench_emb_np):.6f}")
        print(f"ToolBench embedding sample values: {toolbench_emb_np[0][:5]}")
        
        # Trained model
        with torch.no_grad():
            trained_emb = trained_model.encode([query], convert_to_tensor=True, device=device)
            trained_emb_norm = F.normalize(trained_emb, p=2, dim=-1)
            trained_emb_np = trained_emb_norm.cpu().numpy()
        
        print(f"Trained embedding shape: {trained_emb_np.shape}")
        print(f"Trained embedding norm: {np.linalg.norm(trained_emb_np):.6f}")
        print(f"Trained embedding sample values: {trained_emb_np[0][:5]}")
        
        # Compare embeddings
        cosine_sim = F.cosine_similarity(toolbench_emb, trained_emb, dim=-1)
        print(f"Cosine similarity between models: {cosine_sim.item():.6f}")
        
        # Check if embeddings are all zeros or NaN
        if np.all(trained_emb_np == 0):
            print("⚠️  WARNING: Trained model embedding is all zeros!")
        elif np.any(np.isnan(trained_emb_np)):
            print("⚠️  WARNING: Trained model embedding contains NaN values!")
        else:
            print("✅ Trained model embedding looks normal")
    
    # Test model configurations
    print("\n📊 Model Configuration Comparison:")
    print(f"ToolBench model max_seq_length: {toolbench_model.get_max_seq_length()}")
    print(f"Trained model max_seq_length: {trained_model.get_max_seq_length()}")
    
    print(f"ToolBench model dimension: {toolbench_model.get_sentence_embedding_dimension()}")
    print(f"Trained model dimension: {trained_model.get_sentence_embedding_dimension()}")
    
    # Test tokenization
    print("\n🔍 Testing tokenization...")
    test_query = "I need to check the health of the SQUAKE API"
    
    toolbench_tokens = toolbench_model.tokenizer(test_query, return_tensors="pt", padding=True, truncation=True)
    trained_tokens = trained_model.tokenizer(test_query, return_tensors="pt", padding=True, truncation=True)
    
    print(f"ToolBench tokens shape: {toolbench_tokens['input_ids'].shape}")
    print(f"Trained tokens shape: {trained_tokens['input_ids'].shape}")
    
    print(f"ToolBench token IDs: {toolbench_tokens['input_ids'][0][:10].tolist()}")
    print(f"Trained token IDs: {trained_tokens['input_ids'][0][:10].tolist()}")

if __name__ == "__main__":
    debug_embeddings() 