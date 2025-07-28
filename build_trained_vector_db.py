#!/usr/bin/env python3
"""
Build vector database for trained retriever model
"""

import sys
import os
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import faiss
import pickle
from tqdm import tqdm

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_MODEL_PATH = os.path.join(BASE_DIR, "trained_toolbench_retriever_best")
OUTPUT_DIR = os.path.join(BASE_DIR, "vector_db_trained")

def build_trained_vector_database():
    """Build vector database using trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load trained model
    print("\n📦 Loading trained model...")
    trained_model = SentenceTransformer(BEST_MODEL_PATH)
    trained_model = trained_model.to(device)
    trained_model.eval()
    print(f"✅ Trained model loaded")
    
    # Load API data from existing vector database
    print("\n📦 Loading API data...")
    toolbench_db_path = os.path.join(BASE_DIR, "vector_db_toolbench")
    metadata_path = os.path.join(toolbench_db_path, "api_metadata_toolbench.pkl")
    
    with open(metadata_path, 'rb') as f:
        api_metadata = pickle.load(f)
    
    print(f"✅ Loaded {len(api_metadata)} APIs")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate embeddings for all APIs
    print("\n🧪 Generating embeddings for all APIs...")
    api_texts = []
    api_info_list = []
    
    for api_info in tqdm(api_metadata, desc="Processing APIs"):
        tool_name = api_info.get('tool_name', '')
        api_name = api_info.get('api_name', '')
        api_description = api_info.get('api_description', '')
        
        # Create API text representation
        api_text = f"{tool_name} {api_name} {api_description}"
        api_texts.append(api_text)
        api_info_list.append(api_info)
    
    # Generate embeddings in batches
    batch_size = 32
    all_embeddings = []
    
    for i in tqdm(range(0, len(api_texts), batch_size), desc="Generating embeddings"):
        batch_texts = api_texts[i:i+batch_size]
        
        with torch.no_grad():
            embeddings = trained_model.encode(batch_texts, convert_to_tensor=True, device=device)
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            all_embeddings.append(embeddings.cpu().numpy())
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    print(f"✅ Generated embeddings shape: {all_embeddings.shape}")
    
    # Build FAISS index
    print("\n🔧 Building FAISS index...")
    dimension = all_embeddings.shape[1]
    
    # Create index
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    index.add(all_embeddings.astype('float32'))
    
    print(f"✅ Built FAISS index with {index.ntotal} vectors")
    
    # Save index and metadata
    print("\n💾 Saving vector database...")
    
    # Save FAISS index
    index_path = os.path.join(OUTPUT_DIR, "api_embeddings_trained.index")
    faiss.write_index(index, index_path)
    print(f"✅ Saved FAISS index to {index_path}")
    
    # Save metadata
    metadata_path = os.path.join(OUTPUT_DIR, "api_metadata_trained.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(api_info_list, f)
    print(f"✅ Saved metadata to {metadata_path}")
    
    # Save metadata as JSON for inspection
    json_metadata_path = os.path.join(OUTPUT_DIR, "api_metadata_trained.json")
    with open(json_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(api_info_list, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved JSON metadata to {json_metadata_path}")
    
    # Test the index
    print("\n🧪 Testing the built index...")
    test_query = "I need to check the health of the SQUAKE API and get project information"
    
    with torch.no_grad():
        query_embedding = trained_model.encode([test_query], convert_to_tensor=True, device=device)
        query_embedding = F.normalize(query_embedding, p=2, dim=-1)
        query_embedding_np = query_embedding.cpu().numpy()
    
    scores, indices = index.search(query_embedding_np, k=5)
    
    print(f"Test query: '{test_query}'")
    print("Top 5 results:")
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        api_info = api_info_list[idx]
        tool_name = api_info.get('tool_name', 'Unknown')
        api_name = api_info.get('api_name', 'Unknown')
        print(f"  {i+1}. {tool_name}/{api_name} (score: {score:.4f})")
    
    print(f"\n✅ Vector database built successfully!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📊 Total APIs: {len(api_info_list)}")
    print(f"📊 Embedding dimension: {dimension}")

if __name__ == "__main__":
    build_trained_vector_database() 