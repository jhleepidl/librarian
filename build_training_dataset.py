#!/usr/bin/env python3
"""
Build training dataset for query embedding model with L2 normalization
Extract queries from G1, G2, G3 files and create training data with query text and L2-normalized relevant APIs vector sum as label
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import random
from tqdm import tqdm
import pickle

# Configuration
TOOLBENCH_MODEL_NAME = "ToolBench/ToolBench_IR_bert_based_uncased"
DATA_DIR = "../ToolBench/data/instruction"
OUTPUT_DIR = "training_data"
TRAIN_RATIO = 0.9  # 9:1 train:val split

class TrainingDatasetBuilderNormalized:
    def __init__(self):
        """Initialize the dataset builder"""
        self.model = SentenceTransformer(TOOLBENCH_MODEL_NAME)
        self.api_embeddings_cache = {}
        
    def extract_api_text(self, api: Dict[str, Any]) -> str:
        """Convert API information to text format using ToolBench's method"""
        # Use ToolBench's process_retrieval_ducoment format
        category_name = api.get('category_name', '') or ''
        tool_name = api.get('tool_name', '') or ''
        api_name = api.get('api_name', '') or ''
        api_description = api.get('api_description', '') or ''
        required_parameters = json.dumps(api.get('required_parameters', ''))
        optional_parameters = json.dumps(api.get('optional_parameters', ''))
        template_response = json.dumps(api.get('template_response', ''))
        
        # Format exactly like ToolBench's process_retrieval_ducoment function
        text = f"{category_name}, {tool_name}, {api_name}, {api_description}, required_params: {required_parameters}, optional_params: {optional_parameters}, return_schema: {template_response}"
        
        return text
    
    def get_api_embedding(self, api: Dict[str, Any]) -> np.ndarray:
        """Get or create embedding for an API"""
        api_text = self.extract_api_text(api)
        
        # Create API identifier
        api_id = f"{api.get('tool_name', '')}_{api.get('api_name', '')}"
        
        if api_id not in self.api_embeddings_cache:
            self.api_embeddings_cache[api_id] = self.model.encode(api_text)
            
        return self.api_embeddings_cache[api_id]
    
    def get_relevant_apis_sum_embedding(self, query_data: Dict[str, Any]) -> np.ndarray:
        """Get the sum of L2-normalized embeddings for all relevant APIs"""
        if 'relevant APIs' not in query_data or not query_data['relevant APIs']:
            return np.zeros(768)  # Return zero vector if no relevant APIs
            
        relevant_apis = query_data['relevant APIs']
        api_embeddings = []
        
        for api_ref in relevant_apis:
            # api_ref is [tool_name, api_name] format
            if len(api_ref) != 2:
                continue
                
            tool_name, api_name = api_ref
            
            # Find the corresponding API in api_list
            target_api = None
            for api in query_data.get('api_list', []):
                if api.get('tool_name') == tool_name and api.get('api_name') == api_name:
                    target_api = api
                    break
            
            if target_api:
                embedding = self.get_api_embedding(target_api)
                # L2 normalize each API embedding before adding
                norm = np.linalg.norm(embedding)
                if norm > 0:  # Avoid division by zero
                    embedding = embedding / norm
                api_embeddings.append(embedding)
        
        if api_embeddings:
            # Sum all L2-normalized relevant API embeddings
            return np.sum(api_embeddings, axis=0)
        else:
            return np.zeros(768)
    
    def load_query_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load and process a query file"""
        print(f"Loading {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed_data = []
        
        for query_data in tqdm(data, desc=f"Processing {os.path.basename(file_path)}"):
            if 'query' not in query_data:
                continue
                
            query_text = query_data['query']
            
            # Get the sum of L2-normalized relevant APIs embeddings
            relevant_apis_sum = self.get_relevant_apis_sum_embedding(query_data)
            
            # Create training sample
            sample = {
                'query_text': query_text,
                'label_vector': relevant_apis_sum.tolist(),  # Convert to list for JSON serialization
                'relevant_apis_count': len(query_data.get('relevant APIs', [])),
                'api_list_count': len(query_data.get('api_list', []))
            }
            
            processed_data.append(sample)
        
        return processed_data
    
    def build_dataset(self):
        """Build the complete training dataset"""
        print("🔨 Building L2-normalized training dataset...")
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Load all query files
        query_files = [
            os.path.join(DATA_DIR, "G1_query.json"),
            os.path.join(DATA_DIR, "G2_query.json"),
            os.path.join(DATA_DIR, "G3_query.json")
        ]
        
        all_data = []
        
        for file_path in query_files:
            if os.path.exists(file_path):
                file_data = self.load_query_file(file_path)
                all_data.extend(file_data)
                print(f"✅ Loaded {len(file_data)} samples from {os.path.basename(file_path)}")
            else:
                print(f"⚠️  File not found: {file_path}")
        
        print(f"📊 Total samples: {len(all_data)}")
        
        # Shuffle the data
        random.shuffle(all_data)
        
        # Split into train and validation
        split_idx = int(len(all_data) * TRAIN_RATIO)
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
        
        print(f"📈 Train samples: {len(train_data)}")
        print(f"📉 Validation samples: {len(val_data)}")
        
        # Save datasets
        train_path = os.path.join(OUTPUT_DIR, "train_dataset.json")
        val_path = os.path.join(OUTPUT_DIR, "val_dataset.json")
        
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        # Save API embeddings cache for later use
        cache_path = os.path.join(OUTPUT_DIR, "api_embeddings_cache.pkl")
        with open(cache_path, 'wb') as f:
            pickle.dump(self.api_embeddings_cache, f)
        
        # Generate statistics
        self.generate_statistics(train_data, val_data)
        
        print(f"✅ L2-normalized dataset saved to {OUTPUT_DIR}")
        print(f"📁 Train: {train_path}")
        print(f"📁 Validation: {val_path}")
        print(f"📁 API embeddings cache: {cache_path}")
    
    def generate_statistics(self, train_data: List[Dict[str, Any]], val_data: List[Dict[str, Any]]):
        """Generate and save dataset statistics"""
        # Handle empty datasets
        def safe_stats(data_list, key):
            if not data_list:
                return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
            values = [s[key] for s in data_list]
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': min(values),
                'max': max(values)
            }
        
        # Calculate vector norm statistics
        def vector_norm_stats(data_list):
            if not data_list:
                return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
            norms = [np.linalg.norm(np.array(s['label_vector'])) for s in data_list]
            return {
                'mean': np.mean(norms),
                'std': np.std(norms),
                'min': np.min(norms),
                'max': np.max(norms)
            }
        
        stats = {
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'total_samples': len(train_data) + len(val_data),
            'train_relevant_apis_stats': safe_stats(train_data, 'relevant_apis_count'),
            'val_relevant_apis_stats': safe_stats(val_data, 'relevant_apis_count'),
            'train_api_list_stats': safe_stats(train_data, 'api_list_count'),
            'val_api_list_stats': safe_stats(val_data, 'api_list_count'),
            'train_vector_norm_stats': vector_norm_stats(train_data),
            'val_vector_norm_stats': vector_norm_stats(val_data)
        }
        
        stats_path = os.path.join(OUTPUT_DIR, "dataset_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"📊 Statistics saved to {stats_path}")
        
        # Print summary
        print("\n📈 L2-Normalized Dataset Summary:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Train samples: {stats['train_samples']}")
        print(f"  Validation samples: {stats['val_samples']}")
        print(f"  Train/Val ratio: {stats['train_samples']/stats['total_samples']:.2f}")
        print(f"  Average relevant APIs per query (train): {stats['train_relevant_apis_stats']['mean']:.2f}")
        print(f"  Average relevant APIs per query (val): {stats['val_relevant_apis_stats']['mean']:.2f}")
        print(f"  Train vector norm - Mean: {stats['train_vector_norm_stats']['mean']:.4f}, Std: {stats['train_vector_norm_stats']['std']:.4f}")
        print(f"  Val vector norm - Mean: {stats['val_vector_norm_stats']['mean']:.4f}, Std: {stats['val_vector_norm_stats']['std']:.4f}")

def main():
    """Main function"""
    print("🚀 Starting L2-normalized training dataset construction...")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    builder = TrainingDatasetBuilderNormalized()
    builder.build_dataset()
    
    print("✅ L2-normalized training dataset construction completed!")

if __name__ == "__main__":
    main() 