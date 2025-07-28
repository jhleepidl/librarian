#!/usr/bin/env python3
"""
Test magnitude model search functionality
"""

import sys
import os
import json
import torch
import numpy as np
import faiss
import pickle
from tqdm import tqdm

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db_toolbench")
TEST_INSTRUCTION_DIR = "/home/jhlee/ToolBench/data/test_instruction"


class MagnitudeSearchTester:
    """Test magnitude model search functionality"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vector_db = None
        self.api_metadata = []
        
    def load_vector_database(self):
        """Load vector database"""
        print("📦 Loading vector database...")
        
        try:
            # Load FAISS index
            faiss_index_path = os.path.join(VECTOR_DB_PATH, "api_embeddings_toolbench.index")
            self.vector_db = faiss.read_index(faiss_index_path)
            print(f"✅ Loaded FAISS index with {self.vector_db.ntotal} vectors")
            
            # Load metadata
            metadata_path = os.path.join(VECTOR_DB_PATH, "api_metadata_toolbench.pkl")
            with open(metadata_path, 'rb') as f:
                self.api_metadata = pickle.load(f)
            print(f"✅ Loaded metadata for {len(self.api_metadata)} APIs")
            
            return True
        except Exception as e:
            print(f"❌ Error loading vector database: {e}")
            return False
    
    def search_similar_apis(self, query_embedding: np.ndarray, similarity_threshold: float = 0.5) -> list:
        """Search for similar APIs using FAISS"""
        # Normalize query embedding for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Search in FAISS
        D, I = self.vector_db.search(query_norm.reshape(1, -1), k=100)
        
        similar_apis = []
        for i, (distance, idx) in enumerate(zip(D[0], I[0])):
            if idx < len(self.api_metadata):
                api_info = self.api_metadata[idx]
                similarity_score = 1 - distance  # Convert distance to similarity
                
                if similarity_score >= similarity_threshold:
                    similar_apis.append({
                        **api_info,
                        'similarity_score': similarity_score,
                        'rank': i + 1
                    })
        
        return similar_apis
    
    def load_test_instruction_data(self, instruction_dir: str = TEST_INSTRUCTION_DIR) -> list:
        """Load test instruction data"""
        print(f"📦 Loading test instruction data from: {instruction_dir}")
        
        if not os.path.exists(instruction_dir):
            print(f"❌ Test instruction directory not found: {instruction_dir}")
            return []
        
        all_samples = []
        
        # Find all *_instruction.json files
        instruction_files = []
        for file in os.listdir(instruction_dir):
            if file.endswith('_instruction.json'):
                instruction_files.append(file)
        
        print(f"📁 Found {len(instruction_files)} instruction files")
        
        for file in tqdm(instruction_files[:1], desc="Loading instruction files"):  # Only load first file
            file_path = os.path.join(instruction_dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract samples with queries and relevant APIs
                for sample in data[:5]:  # Only first 5 samples
                    query = sample.get("query", "")
                    api_list = sample.get("api_list", [])
                    relevant_api_keys = set()
                    
                    # Extract relevant APIs
                    for rel in sample.get("relevant APIs", []):
                        if len(rel) == 2:
                            relevant_api_keys.add((rel[0], rel[1]))  # (tool_name, api_name)
                    
                    relevant_apis = [api for api in api_list if (api.get("tool_name"), api.get("api_name")) in relevant_api_keys]
                    
                    if query and relevant_apis:  # Only include samples with both query and relevant APIs
                        all_samples.append({
                            "query": query,
                            "relevant_apis": relevant_apis,
                            "api_list": api_list,
                            "query_id": sample.get("query_id", None)
                        })
                            
            except Exception as e:
                print(f"⚠️  Error loading {file}: {e}")
                continue
        
        print(f"✅ Loaded {len(all_samples)} samples from test instruction data")
        return all_samples
    
    def test_search_with_dummy_embeddings(self, test_samples: list):
        """Test search with dummy embeddings"""
        print("\n🔍 Testing search with dummy embeddings...")
        print("=" * 80)
        
        for i, sample in enumerate(test_samples[:3]):  # Test first 3 samples
            query = sample.get('query', '')
            relevant_apis = sample.get('relevant_apis', [])
            
            print(f"\n📝 Query {i+1}: {query[:100]}...")
            print(f"   Relevant APIs: {len(relevant_apis)}")
            
            # Create dummy embedding (random vector)
            dummy_embedding = np.random.randn(768)  # 768-dim vector
            dummy_embedding = dummy_embedding / np.linalg.norm(dummy_embedding)  # Normalize
            
            # Search with different thresholds
            thresholds = [0.1, 0.3, 0.5, 0.7]
            
            for threshold in thresholds:
                similar_apis = self.search_similar_apis(dummy_embedding, similarity_threshold=threshold)
                
                print(f"\n   Threshold {threshold}:")
                print(f"     Found {len(similar_apis)} APIs")
                
                # Check if any relevant APIs are found
                found_relevant = 0
                for relevant_api in relevant_apis:
                    for found_api in similar_apis:
                        if (found_api['tool_name'] == relevant_api['tool_name'] and 
                            found_api['api_name'] == relevant_api['api_name']):
                            found_relevant += 1
                            break
                
                print(f"     Relevant APIs found: {found_relevant}/{len(relevant_apis)}")
                
                # Show top 3 results
                if similar_apis:
                    print(f"     Top 3 results:")
                    for j, api in enumerate(similar_apis[:3]):
                        print(f"       {j+1}. {api['tool_name']} - {api['api_name']} (sim: {api['similarity_score']:.3f})")
            
            print("-" * 80)
    
    def analyze_search_behavior(self):
        """Analyze search behavior with different embeddings"""
        print("\n🔍 Analyzing search behavior...")
        print("=" * 50)
        
        # Test with different types of embeddings
        test_cases = [
            ("Random embedding", np.random.randn(768)),
            ("Zero embedding", np.zeros(768)),
            ("Ones embedding", np.ones(768)),
            ("Unit vector", np.array([1.0] + [0.0] * 767))
        ]
        
        for name, embedding in test_cases:
            print(f"\n📊 Testing {name}:")
            
            # Normalize embedding
            if np.linalg.norm(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)
            
            # Search with different thresholds
            thresholds = [0.1, 0.3, 0.5, 0.7]
            
            for threshold in thresholds:
                similar_apis = self.search_similar_apis(embedding, similarity_threshold=threshold)
                print(f"   Threshold {threshold}: {len(similar_apis)} APIs found")
                
                if similar_apis:
                    top_similarity = similar_apis[0]['similarity_score']
                    print(f"     Top similarity: {top_similarity:.4f}")
    
    def run_test(self):
        """Run the complete test"""
        print("🚀 Starting Magnitude Search Test")
        print("=" * 50)
        
        # Load vector database
        if not self.load_vector_database():
            print("❌ Failed to load vector database")
            return
        
        # Load test data
        test_samples = self.load_test_instruction_data()
        
        if not test_samples:
            print("❌ No test samples found")
            return
        
        # Test search with dummy embeddings
        self.test_search_with_dummy_embeddings(test_samples)
        
        # Analyze search behavior
        self.analyze_search_behavior()
        
        print("\n✅ Magnitude search test completed!")


def main():
    """Main test function"""
    tester = MagnitudeSearchTester()
    tester.run_test()


if __name__ == "__main__":
    main() 