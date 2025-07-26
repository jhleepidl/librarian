#!/usr/bin/env python3
"""
Check similarity between API embeddings using trained model
"""

import sys
import os
import json
import torch
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from tqdm import tqdm

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINED_MODEL_PATH = os.path.join(BASE_DIR, "trained_toolbench_retriever_best")
TEST_INSTRUCTION_DIR = "/home/jhlee/ToolBench/data/test_instruction"


class EmbeddingSimilarityChecker:
    """
    Check similarity between API embeddings
    """
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        """Load the trained model"""
        print(f"📦 Loading trained model from: {TRAINED_MODEL_PATH}")
        
        try:
            self.model = SentenceTransformer(TRAINED_MODEL_PATH)
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded successfully on {self.device}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def load_sample_apis(self, num_apis: int = 20) -> List[Dict[str, Any]]:
        """Load a sample of APIs from test instruction data"""
        print(f"📦 Loading sample APIs from: {TEST_INSTRUCTION_DIR}")
        
        if not os.path.exists(TEST_INSTRUCTION_DIR):
            print(f"❌ Test instruction directory not found: {TEST_INSTRUCTION_DIR}")
            return []
        
        all_apis = []
        api_set = set()
        
        # Find all *_instruction.json files
        instruction_files = []
        for file in os.listdir(TEST_INSTRUCTION_DIR):
            if file.endswith('_instruction.json'):
                instruction_files.append(file)
        
        print(f"📁 Found {len(instruction_files)} instruction files")
        
        for file in instruction_files[:2]:  # Use first 2 files for speed
            file_path = os.path.join(TEST_INSTRUCTION_DIR, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract APIs from the data
                for sample in data:
                    api_list = sample.get("api_list", [])
                    for api in api_list:
                        # Create unique identifier for API
                        api_id = (api.get("tool_name", ""), api.get("api_name", ""))
                        if api_id not in api_set:
                            api_set.add(api_id)
                            all_apis.append(api)
                            
                            if len(all_apis) >= num_apis:
                                break
                    if len(all_apis) >= num_apis:
                        break
                if len(all_apis) >= num_apis:
                    break
                            
            except Exception as e:
                print(f"⚠️  Error loading {file}: {e}")
                continue
        
        print(f"✅ Loaded {len(all_apis)} sample APIs")
        return all_apis
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts using the trained model
        Use the same approach as during training
        """
        if not self.model:
            raise ValueError("Model not loaded")
        
        # Use the same approach as during training
        tokenizer = self.model.tokenizer
        max_length = self.model.get_max_seq_length()
        features = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        features = {k: v.to(self.device) for k, v in features.items()}
        
        with torch.no_grad():
            out = self.model({"input_ids": features["input_ids"], "attention_mask": features["attention_mask"]})
            embeddings = out["sentence_embedding"]
        
        return embeddings.cpu().numpy()
    
    def create_api_embeddings(self, apis: List[Dict[str, Any]]) -> tuple:
        """Create embeddings for APIs"""
        print("🔧 Creating API embeddings...")
        
        # Prepare API texts with detailed information
        api_texts = []
        api_info = []
        
        for api in apis:
            tool_name = api.get('tool_name', '')
            api_name = api.get('api_name', '')
            api_description = api.get('api_description', '')
            category = api.get('category_name', '')
            method = api.get('method', '')
            
            # Create detailed API text
            api_text = f"Tool: {tool_name}, API: {api_name}, Category: {category}, Method: {method}, Description: {api_description}"
            api_texts.append(api_text)
            
            api_info.append({
                'tool_name': tool_name,
                'api_name': api_name,
                'category': category,
                'method': method,
                'description': api_description[:100] + "..." if len(api_description) > 100 else api_description
            })
        
        # Generate embeddings
        embeddings = self.get_embeddings(api_texts)
        print(f"✅ Generated embeddings shape: {embeddings.shape}")
        
        return embeddings, api_info
    
    def analyze_similarities(self, embeddings: np.ndarray, api_info: List[Dict[str, Any]]):
        """Analyze similarities between API embeddings"""
        print("\n📊 Analyzing API embedding similarities...")
        
        # Normalize embeddings for cosine similarity
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Calculate similarity matrix
        similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)
        
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
        print(f"Diagonal values (should be 1.0): {np.diag(similarity_matrix)}")
        
        # Remove diagonal (self-similarity)
        similarity_no_diag = similarity_matrix[~np.eye(similarity_matrix.shape[0], dtype=bool)]
        
        print(f"\n📈 Similarity Statistics:")
        print(f"  Mean: {np.mean(similarity_no_diag):.6f}")
        print(f"  Std: {np.std(similarity_no_diag):.6f}")
        print(f"  Min: {np.min(similarity_no_diag):.6f}")
        print(f"  Max: {np.max(similarity_no_diag):.6f}")
        print(f"  > 0.9: {np.sum(similarity_no_diag > 0.9)} / {len(similarity_no_diag)}")
        print(f"  > 0.95: {np.sum(similarity_no_diag > 0.95)} / {len(similarity_no_diag)}")
        print(f"  > 0.99: {np.sum(similarity_no_diag > 0.99)} / {len(similarity_no_diag)}")
        
        # Find highest similarities
        print(f"\n🔍 Top 10 Highest Similarities:")
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = similarity_matrix[i, j]
                if sim > 0.8:  # Show only high similarities
                    api1 = api_info[i]
                    api2 = api_info[j]
                    print(f"  {sim:.4f}: {api1['tool_name']} - {api1['api_name']} <-> {api2['tool_name']} - {api2['api_name']}")
        
        # Show some examples
        print(f"\n📋 Sample API Embeddings:")
        for i, info in enumerate(api_info[:5]):
            print(f"  {i+1}. {info['tool_name']} - {info['api_name']}")
            print(f"     Category: {info['category']}")
            print(f"     Method: {info['method']}")
            print(f"     Description: {info['description']}")
            print()
    
    def run_analysis(self, num_apis: int = 20):
        """Run the complete analysis"""
        print("🔍 API Embedding Similarity Analysis")
        print("=" * 50)
        
        # Load model
        if not self.load_model():
            return False
        
        # Load sample APIs
        apis = self.load_sample_apis(num_apis)
        if not apis:
            return False
        
        # Create embeddings
        embeddings, api_info = self.create_api_embeddings(apis)
        
        # Analyze similarities
        self.analyze_similarities(embeddings, api_info)
        
        print("\n✅ Analysis completed!")


def main():
    """Main function"""
    checker = EmbeddingSimilarityChecker()
    checker.run_analysis(num_apis=20)


if __name__ == "__main__":
    main() 