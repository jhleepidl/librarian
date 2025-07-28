#!/usr/bin/env python3
"""
Test vector database compatibility with different models
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

def test_vector_db_compatibility():
    """Test which vector database works with which model"""
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
    
    # Test query
    test_query = "I need to check the health of the SQUAKE API and get project information"
    
    # Get embeddings
    with torch.no_grad():
        toolbench_emb = toolbench_model.encode([test_query], convert_to_tensor=True, device=device)
        toolbench_emb_norm = F.normalize(toolbench_emb, p=2, dim=-1)
        toolbench_emb_np = toolbench_emb_norm.cpu().numpy()
        
        trained_emb = trained_model.encode([test_query], convert_to_tensor=True, device=device)
        trained_emb_norm = F.normalize(trained_emb, p=2, dim=-1)
        trained_emb_np = trained_emb_norm.cpu().numpy()
    
    # Test different vector databases
    vector_dbs = [
        ("vector_db_toolbench", "api_embeddings_toolbench.index", "api_metadata_toolbench.pkl"),
        ("vector_db", "api_embeddings.index", "api_metadata.pkl"),
        ("faiss_database", "faiss_index.bin", "api_info.json"),
        ("faiss_database_l2", "faiss_index.bin", "api_info.json")
    ]
    
    print("\n🧪 Testing vector database compatibility...")
    
    for db_name, index_file, metadata_file in vector_dbs:
        print(f"\n--- Testing {db_name} ---")
        
        try:
            # Load vector database
            index_path = os.path.join(BASE_DIR, db_name, index_file)
            vector_db = faiss.read_index(index_path)
            
            # Load metadata
            metadata_path = os.path.join(BASE_DIR, db_name, metadata_file)
            if metadata_file.endswith('.pkl'):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
            else:
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            print(f"✅ Loaded {db_name}: {vector_db.ntotal} vectors, {vector_db.d} dimensions")
            
            # Test ToolBench model
            print("🔍 Testing ToolBench model:")
            scores_tb, indices_tb = vector_db.search(toolbench_emb_np, k=3)
            print(f"  Top 3 scores: {scores_tb[0]}")
            
            # Show results
            for j, (score, idx) in enumerate(zip(scores_tb[0], indices_tb[0])):
                if isinstance(metadata, list):
                    api_info = metadata[idx]
                else:
                    api_info = metadata[str(idx)]
                tool_name = api_info.get('tool_name', 'Unknown')
                api_name = api_info.get('api_name', 'Unknown')
                print(f"  {j+1}. {tool_name}/{api_name} (score: {score:.4f})")
            
            # Test trained model
            print("🔍 Testing trained model:")
            scores_tr, indices_tr = vector_db.search(trained_emb_np, k=3)
            print(f"  Top 3 scores: {scores_tr[0]}")
            
            # Show results
            for j, (score, idx) in enumerate(zip(scores_tr[0], indices_tr[0])):
                if isinstance(metadata, list):
                    api_info = metadata[idx]
                else:
                    api_info = metadata[str(idx)]
                tool_name = api_info.get('tool_name', 'Unknown')
                api_name = api_info.get('api_name', 'Unknown')
                print(f"  {j+1}. {tool_name}/{api_name} (score: {score:.4f})")
            
            # Determine which model works better
            avg_score_tb = np.mean(scores_tb[0])
            avg_score_tr = np.mean(scores_tr[0])
            
            print(f"📊 Average scores - ToolBench: {avg_score_tb:.4f}, Trained: {avg_score_tr:.4f}")
            
            if avg_score_tb > avg_score_tr:
                print("✅ This database works better with ToolBench model")
            elif avg_score_tr > avg_score_tb:
                print("✅ This database works better with trained model")
            else:
                print("🤝 Both models work similarly")
                
        except Exception as e:
            print(f"❌ Error with {db_name}: {e}")

if __name__ == "__main__":
    test_vector_db_compatibility() 