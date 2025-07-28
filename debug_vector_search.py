#!/usr/bin/env python3
"""
Debug vector database search with trained model
"""

import sys
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import faiss
import pickle

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINED_MODEL_PATH = os.path.join(BASE_DIR, "trained_toolbench_retriever")
BEST_MODEL_PATH = os.path.join(BASE_DIR, "trained_toolbench_retriever_best")
TOOLBENCH_MODEL_NAME = "ToolBench/ToolBench_IR_bert_based_uncased"
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db_toolbench")

def debug_vector_search():
    """Debug vector database search with both models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    print("\n📦 Loading models...")
    toolbench_model = SentenceTransformer(TOOLBENCH_MODEL_NAME)
    toolbench_model = toolbench_model.to(device)
    toolbench_model.eval()
    
    trained_model = SentenceTransformer(BEST_MODEL_PATH)
    trained_model = trained_model.to(device)
    trained_model.eval()
    
    # Load vector database
    print("\n📦 Loading vector database...")
    faiss_index_path = os.path.join(VECTOR_DB_PATH, "api_embeddings_toolbench.index")
    vector_db = faiss.read_index(faiss_index_path)
    
    metadata_path = os.path.join(VECTOR_DB_PATH, "api_metadata_toolbench.pkl")
    with open(metadata_path, 'rb') as f:
        api_metadata = pickle.load(f)
    
    print(f"✅ Loaded FAISS index with {vector_db.ntotal} vectors")
    print(f"✅ Loaded metadata for {len(api_metadata)} APIs")
    
    # Test queries
    test_queries = [
        "I need to check the health of the SQUAKE API and get project information",
        "I want to track a package through Correo Argentino",
        "I'm looking for cocktail recipes and financial data"
    ]
    
    print("\n🧪 Testing vector search...")
    
    for i, query in enumerate(test_queries):
        print(f"\n--- Query {i+1}: '{query}' ---")
        
        # Get embeddings
        with torch.no_grad():
            toolbench_emb = toolbench_model.encode([query], convert_to_tensor=True, device=device)
            toolbench_emb_norm = F.normalize(toolbench_emb, p=2, dim=-1)
            toolbench_emb_np = toolbench_emb_norm.cpu().numpy()
            
            trained_emb = trained_model.encode([query], convert_to_tensor=True, device=device)
            trained_emb_norm = F.normalize(trained_emb, p=2, dim=-1)
            trained_emb_np = trained_emb_norm.cpu().numpy()
        
        # Search with ToolBench model
        print("🔍 ToolBench model search:")
        scores_tb, indices_tb = vector_db.search(toolbench_emb_np, k=5)
        print(f"  Top 5 scores: {scores_tb[0]}")
        print(f"  Top 5 indices: {indices_tb[0]}")
        
        # Show top results
        for j, (score, idx) in enumerate(zip(scores_tb[0], indices_tb[0])):
            api_info = api_metadata[idx]
            print(f"  {j+1}. {api_info['tool_name']}/{api_info['api_name']} (score: {score:.4f})")
        
        # Search with trained model
        print("🔍 Trained model search:")
        scores_tr, indices_tr = vector_db.search(trained_emb_np, k=5)
        print(f"  Top 5 scores: {scores_tr[0]}")
        print(f"  Top 5 indices: {indices_tr[0]}")
        
        # Show top results
        for j, (score, idx) in enumerate(zip(scores_tr[0], indices_tr[0])):
            api_info = api_metadata[idx]
            print(f"  {j+1}. {api_info['tool_name']}/{api_info['api_name']} (score: {score:.4f})")
        
        # Compare results
        print("📊 Result comparison:")
        toolbench_apis = set(api_metadata[idx]['tool_name'] for idx in indices_tb[0])
        trained_apis = set(api_metadata[idx]['tool_name'] for idx in indices_tr[0])
        overlap = toolbench_apis.intersection(trained_apis)
        print(f"  ToolBench APIs: {toolbench_apis}")
        print(f"  Trained APIs: {trained_apis}")
        print(f"  Overlap: {overlap}")
        print(f"  Overlap ratio: {len(overlap)/len(toolbench_apis):.2f}")
        
        # Check if trained model scores are all very low
        if np.all(scores_tr[0] < 0.1):
            print("⚠️  WARNING: Trained model scores are all very low!")
        else:
            print("✅ Trained model scores look reasonable")

if __name__ == "__main__":
    debug_vector_search() 