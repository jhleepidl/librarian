#!/usr/bin/env python3
"""
Test the trained query embedding model
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
TEST_INSTRUCTION_DIR = "/home/jhlee/ToolBench/data/test_instruction"
QUERY_MODEL_PATH = "/home/jhlee/librarian/trained_query_model/best_model.pt"
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db_query")


class QueryEmbeddingModel(torch.nn.Module):
    """
    Query embedding model that learns to map queries to API embedding space
    """
    
    def __init__(self, base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        # Use a pre-trained model as the base
        self.base_model = SentenceTransformer(base_model_name)
        
        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Add a projection layer to map to the same dimension as API embeddings
        self.projection = torch.nn.Linear(384, 768)  # all-MiniLM-L6-v2 -> ToolBench dimension
        
        # Add a small MLP for fine-tuning
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(768, 768)
        )
    
    def forward(self, input_ids, attention_mask):
        # Get base embeddings
        with torch.no_grad():
            base_output = self.base_model({"input_ids": input_ids, "attention_mask": attention_mask})
            base_embeddings = base_output["sentence_embedding"]
        
        # Project to target dimension
        projected = self.projection(base_embeddings)
        
        # Apply MLP
        output = self.mlp(projected)
        
        return output


class QueryEmbeddingTester:
    """
    Test the trained query embedding model
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.query_model = None
        self.api_model = None
        self.vector_db = None
        self.api_metadata = []
        
    def load_models(self):
        """Load trained query model and API model"""
        print("📦 Loading models...")
        
        # Load trained query model
        self.query_model = QueryEmbeddingModel()
        if os.path.exists(QUERY_MODEL_PATH):
            checkpoint = torch.load(QUERY_MODEL_PATH, map_location=self.device)
            self.query_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded trained query model from {QUERY_MODEL_PATH}")
        else:
            print(f"❌ Trained query model not found at {QUERY_MODEL_PATH}")
            return False
        
        self.query_model = self.query_model.to(self.device)
        self.query_model.eval()
        
        # Load API embedding model (ToolBench, frozen)
        self.api_model = SentenceTransformer("ToolBench/ToolBench_IR_bert_based_uncased")
        self.api_model = self.api_model.to(self.device)
        self.api_model.eval()
        print("✅ API embedding model loaded")
        
        return True
    
    def get_query_embedding(self, queries: List[str]) -> np.ndarray:
        """Get query embeddings using the trained model"""
        tokenizer = self.query_model.base_model.tokenizer
        max_length = self.query_model.base_model.get_max_seq_length()
        
        features = tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        features = {k: v.to(self.device) for k, v in features.items()}
        
        with torch.no_grad():
            embeddings = self.query_model(features["input_ids"], features["attention_mask"])
        
        return embeddings.cpu().numpy()
    
    def get_api_embedding(self, api_texts: List[str]) -> np.ndarray:
        """Get API embeddings using the ToolBench model"""
        with torch.no_grad():
            embeddings = self.api_model.encode(api_texts, convert_to_tensor=True, device=self.device)
        return embeddings.cpu().numpy()
    
    def format_api_text(self, api: Dict[str, Any]) -> str:
        """Format API info into text string"""
        tool_name = api.get('tool_name', '')
        api_name = api.get('api_name', '')
        api_description = api.get('api_description', '')
        category = api.get('category_name', '')
        method = api.get('method', '')
        
        return f"Tool: {tool_name}, API: {api_name}, Category: {category}, Method: {method}, Description: {api_description}"
    
    def load_test_instruction_data(self) -> List[Dict[str, Any]]:
        """Load test instruction data"""
        print(f"📦 Loading test instruction data from: {TEST_INSTRUCTION_DIR}")
        
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
        
        for file in tqdm(instruction_files, desc="Loading instruction files"):
            file_path = os.path.join(TEST_INSTRUCTION_DIR, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract all APIs from the data
                for sample in data:
                    api_list = sample.get("api_list", [])
                    for api in api_list:
                        # Create unique identifier for API
                        api_id = (api.get("tool_name", ""), api.get("api_name", ""))
                        if api_id not in api_set:
                            api_set.add(api_id)
                            all_apis.append(api)
                            
            except Exception as e:
                print(f"⚠️  Error loading {file}: {e}")
                continue
        
        print(f"✅ Loaded {len(all_apis)} unique APIs from test instruction data")
        return all_apis
    
    def create_vector_database(self, apis: List[Dict[str, Any]], batch_size: int = 32):
        """Create vector database using API embeddings"""
        print(f"🔧 Creating vector database for {len(apis)} APIs...")
        
        # Create output directory
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        
        # Prepare API texts and metadata
        api_texts = []
        self.api_metadata = []
        
        for api in tqdm(apis, desc="Preparing API data"):
            # Format API text for embedding
            api_text = self.format_api_text(api)
            api_texts.append(api_text)
            
            # Store metadata
            self.api_metadata.append({
                'tool_name': api.get('tool_name', ''),
                'api_name': api.get('api_name', ''),
                'category_name': api.get('category_name', ''),
                'api_description': api.get('api_description', ''),
                'required_parameters': api.get('required_parameters', []),
                'optional_parameters': api.get('optional_parameters', []),
                'method': api.get('method', ''),
                'original_data': api
            })
        
        # Generate embeddings in batches
        all_embeddings = []
        
        for i in tqdm(range(0, len(api_texts), batch_size), desc="Generating embeddings"):
            batch_texts = api_texts[i:i + batch_size]
            batch_embeddings = self.get_api_embedding(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        embeddings_array = np.vstack(all_embeddings)
        print(f"✅ Generated embeddings shape: {embeddings_array.shape}")
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        self.vector_db = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Add vectors to index
        self.vector_db.add(embeddings_array)
        print(f"✅ Added {self.vector_db.ntotal} vectors to FAISS index")
        
        # Save vector database and metadata
        self.save_vector_database()
        
        return embeddings_array
    
    def save_vector_database(self):
        """Save vector database and metadata"""
        # Save FAISS index
        faiss_index_path = os.path.join(VECTOR_DB_PATH, "api_embeddings_query.index")
        faiss.write_index(self.vector_db, faiss_index_path)
        print(f"💾 Saved FAISS index to: {faiss_index_path}")
        
        # Save metadata
        metadata_path = os.path.join(VECTOR_DB_PATH, "api_metadata_query.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.api_metadata, f)
        print(f"💾 Saved metadata to: {metadata_path}")
        
        # Save metadata as JSON for easy inspection
        metadata_json_path = os.path.join(VECTOR_DB_PATH, "api_metadata_query.json")
        with open(metadata_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.api_metadata, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved metadata JSON to: {metadata_json_path}")
    
    def search_similar_apis(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar APIs given a query"""
        if not self.vector_db:
            print("❌ Vector database not loaded")
            return []
        
        # Get query embedding using trained model
        query_embedding = self.get_query_embedding([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.vector_db.search(query_embedding, top_k)
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.api_metadata):
                result = {
                    'rank': i + 1,
                    'score': float(score),
                    'api': self.api_metadata[idx]
                }
                results.append(result)
        
        return results
    
    def test_model_with_queries(self, test_queries: List[str] = None):
        """Test the model with sample queries"""
        if test_queries is None:
            test_queries = [
                "I need to check the health of the SQUAKE API and get project information",
                "I want to track a package through Correo Argentino",
                "I'm looking for cocktail recipes and financial data",
                "I need to convert currency and get weather information",
                "I want to find restaurants and book a flight"
            ]
        
        print(f"\n🧪 Testing model with {len(test_queries)} queries...")
        
        for i, query in enumerate(test_queries):
            print(f"\n📝 Query {i+1}: '{query}'")
            
            results = self.search_similar_apis(query, top_k=3)
            
            print("🔍 Top recommendations:")
            for result in results:
                api = result['api']
                print(f"  {result['rank']}. {api['tool_name']} - {api['api_name']} (score: {result['score']:.4f})")
                print(f"     Category: {api['category_name']}")
                print(f"     Description: {api['api_description'][:100]}...")
    
    def run_test(self):
        """Run the complete test"""
        print("🧪 Query Embedding Model Test")
        print("=" * 50)
        
        # Load models
        if not self.load_models():
            return False
        
        # Load test data
        apis = self.load_test_instruction_data()
        if not apis:
            return False
        
        # Create vector database
        embeddings = self.create_vector_database(apis)
        
        # Test with sample queries
        self.test_model_with_queries()
        
        print("\n✅ Test completed!")
        return True


def main():
    """Main test function"""
    tester = QueryEmbeddingTester()
    tester.run_test()


if __name__ == "__main__":
    main() 