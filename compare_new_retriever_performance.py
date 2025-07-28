#!/usr/bin/env python3
"""
Compare performance between ToolBench model and newly trained retriever model
"""

import sys
import os
import json
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Set
import logging
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from tqdm import tqdm
import time
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINED_MODEL_PATH = os.path.join(BASE_DIR, "trained_toolbench_retriever")
BEST_MODEL_PATH = os.path.join(BASE_DIR, "trained_toolbench_retriever_best")
TOOLBENCH_MODEL_NAME = "ToolBench/ToolBench_IR_bert_based_uncased"
TOOLBENCH_VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db_toolbench")
TRAINED_VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db_trained")
TEST_INSTRUCTION_DIR = "/home/jhlee/ToolBench/data/test_instruction"


class NewRetrieverPerformanceComparator:
    """
    Compare performance between ToolBench model and newly trained retriever model
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.toolbench_model = None
        self.trained_model = None
        self.toolbench_vector_db = None
        self.trained_vector_db = None
        self.toolbench_api_metadata = []
        self.trained_api_metadata = []
        
    def load_models(self):
        """Load both ToolBench and trained models"""
        print("📦 Loading models...")
        
        # Load ToolBench model
        try:
            self.toolbench_model = SentenceTransformer(TOOLBENCH_MODEL_NAME)
            self.toolbench_model = self.toolbench_model.to(self.device)
            self.toolbench_model.eval()
            print(f"✅ ToolBench model loaded")
        except Exception as e:
            print(f"❌ Error loading ToolBench model: {e}")
            return False
        
        # Load trained retriever model
        try:
            # Try to load the best model first, then fallback to final model
            if os.path.exists(BEST_MODEL_PATH):
                self.trained_model = SentenceTransformer(BEST_MODEL_PATH)
                print(f"✅ Trained retriever model loaded from {BEST_MODEL_PATH}")
            elif os.path.exists(TRAINED_MODEL_PATH):
                self.trained_model = SentenceTransformer(TRAINED_MODEL_PATH)
                print(f"✅ Trained retriever model loaded from {TRAINED_MODEL_PATH}")
            else:
                print(f"⚠️  No trained model found at {BEST_MODEL_PATH} or {TRAINED_MODEL_PATH}")
                return False
            
            self.trained_model = self.trained_model.to(self.device)
            self.trained_model.eval()
        except Exception as e:
            print(f"❌ Error loading trained model: {e}")
            return False
        
        return True
    
    def load_vector_database(self):
        """Load vector databases for both models"""
        print("📦 Loading vector databases...")
        
        try:
            # Load ToolBench vector database
            toolbench_index_path = os.path.join(TOOLBENCH_VECTOR_DB_PATH, "api_embeddings_toolbench.index")
            self.toolbench_vector_db = faiss.read_index(toolbench_index_path)
            print(f"✅ Loaded ToolBench FAISS index with {self.toolbench_vector_db.ntotal} vectors")
            
            toolbench_metadata_path = os.path.join(TOOLBENCH_VECTOR_DB_PATH, "api_metadata_toolbench.pkl")
            with open(toolbench_metadata_path, 'rb') as f:
                self.toolbench_api_metadata = pickle.load(f)
            print(f"✅ Loaded ToolBench metadata for {len(self.toolbench_api_metadata)} APIs")
            
            # Load trained model vector database
            trained_index_path = os.path.join(TRAINED_VECTOR_DB_PATH, "api_embeddings_trained.index")
            trained_metadata_path = os.path.join(TRAINED_VECTOR_DB_PATH, "api_metadata_trained.pkl")
            
            # Check if trained vector database exists, if not build it
            if not os.path.exists(trained_index_path) or not os.path.exists(trained_metadata_path):
                print("⚠️  Trained model vector database not found. Building it...")
                self._build_trained_vector_database()
            
            self.trained_vector_db = faiss.read_index(trained_index_path)
            print(f"✅ Loaded trained model FAISS index with {self.trained_vector_db.ntotal} vectors")
            
            with open(trained_metadata_path, 'rb') as f:
                self.trained_api_metadata = pickle.load(f)
            print(f"✅ Loaded trained model metadata for {len(self.trained_api_metadata)} APIs")
            
            return True
        except Exception as e:
            print(f"❌ Error loading vector databases: {e}")
            return False
    
    def _build_trained_vector_database(self):
        """Build vector database for trained model"""
        print("🔧 Building trained model vector database...")
        
        # Create output directory
        os.makedirs(TRAINED_VECTOR_DB_PATH, exist_ok=True)
        
        # Load API data from ToolBench vector database
        toolbench_db_path = os.path.join(BASE_DIR, "vector_db_toolbench")
        metadata_path = os.path.join(toolbench_db_path, "api_metadata_toolbench.pkl")
        
        with open(metadata_path, 'rb') as f:
            api_metadata = pickle.load(f)
        
        print(f"✅ Loaded {len(api_metadata)} APIs")
        
        # Generate embeddings for all APIs
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
                embeddings = self.trained_model.encode(batch_texts, convert_to_tensor=True, device=self.device)
                # Normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=-1)
                all_embeddings.append(embeddings.cpu().numpy())
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(all_embeddings)
        print(f"✅ Generated embeddings shape: {all_embeddings.shape}")
        
        # Build FAISS index
        dimension = all_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(all_embeddings.astype('float32'))
        
        print(f"✅ Built FAISS index with {index.ntotal} vectors")
        
        # Save index and metadata
        index_path = os.path.join(TRAINED_VECTOR_DB_PATH, "api_embeddings_trained.index")
        faiss.write_index(index, index_path)
        print(f"✅ Saved FAISS index to {index_path}")
        
        metadata_path = os.path.join(TRAINED_VECTOR_DB_PATH, "api_metadata_trained.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(api_info_list, f)
        print(f"✅ Saved metadata to {metadata_path}")
        
        print("✅ Trained model vector database built successfully!")
    
    def get_toolbench_embedding(self, query: str) -> np.ndarray:
        """Get query embedding using ToolBench model"""
        with torch.no_grad():
            embedding = self.toolbench_model.encode([query], convert_to_tensor=True, device=self.device)
            # Normalize for cosine similarity
            normalized_embedding = F.normalize(embedding, p=2, dim=-1)
            return normalized_embedding.cpu().numpy()
    
    def get_trained_embedding(self, query: str) -> np.ndarray:
        """Get query embedding using trained model"""
        with torch.no_grad():
            embedding = self.trained_model.encode([query], convert_to_tensor=True, device=self.device)
            # Normalize for cosine similarity
            normalized_embedding = F.normalize(embedding, p=2, dim=-1)
            return normalized_embedding.cpu().numpy()
    
    def search_similar_apis_toolbench(self, query_embedding: np.ndarray, similarity_threshold: float = 0.5, top_k: int = 100) -> List[Dict[str, Any]]:
        """Search for similar APIs using ToolBench vector database"""
        # Search in ToolBench vector database
        # Normalize query embedding for cosine similarity
        query_norm = np.linalg.norm(query_embedding, axis=-1, keepdims=True)
        query_embedding_normalized = query_embedding / (query_norm + 1e-8)
        scores, indices = self.toolbench_vector_db.search(query_embedding_normalized, k=top_k)
        
        # Filter by similarity threshold
        similar_apis = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= similarity_threshold:
                api_info = self.toolbench_api_metadata[idx].copy()
                api_info['similarity_score'] = float(score)
                similar_apis.append(api_info)
        
        return similar_apis
    
    def search_similar_apis_trained(self, query_embedding: np.ndarray, similarity_threshold: float = 0.5, top_k: int = 100) -> List[Dict[str, Any]]:
        """Search for similar APIs using trained model vector database"""
        # Search in trained model vector database
        # Normalize query embedding for cosine similarity
        query_norm = np.linalg.norm(query_embedding, axis=-1, keepdims=True)
        query_embedding_normalized = query_embedding / (query_norm + 1e-8)
        scores, indices = self.trained_vector_db.search(query_embedding_normalized, k=top_k)
        
        # Filter by similarity threshold
        similar_apis = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= similarity_threshold:
                api_info = self.trained_api_metadata[idx].copy()
                api_info['similarity_score'] = float(score)
                similar_apis.append(api_info)
        
        return similar_apis
    
    def evaluate_predictions(self, predicted_apis: List[Dict[str, Any]], relevant_apis: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate predictions against ground truth"""
        # Create sets of predicted and relevant APIs
        predicted_set = set()
        for api in predicted_apis:
            api_id = (api['tool_name'], api['api_name'])
            predicted_set.add(api_id)
        
        relevant_set = set()
        for api in relevant_apis:
            api_id = (api.get('tool_name', ''), api.get('api_name', ''))
            relevant_set.add(api_id)
        
        # Calculate metrics
        if len(predicted_set) == 0 and len(relevant_set) == 0:
            precision = recall = f1 = 1.0
        elif len(predicted_set) == 0:
            precision = recall = f1 = 0.0
        elif len(relevant_set) == 0:
            precision = recall = f1 = 0.0
        else:
            intersection = predicted_set.intersection(relevant_set)
            precision = len(intersection) / len(predicted_set) if len(predicted_set) > 0 else 0
            recall = len(intersection) / len(relevant_set) if len(relevant_set) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predicted_count': len(predicted_set),
            'relevant_count': len(relevant_set),
            'intersection_count': len(predicted_set.intersection(relevant_set))
        }
    
    def test_toolbench_model(self, test_samples: List[Dict[str, Any]], similarity_threshold: float = 0.5, top_k: int = 100) -> Dict[str, Any]:
        """Test ToolBench model performance"""
        print(f"\n🧪 Testing ToolBench model (threshold={similarity_threshold}, top_k={top_k})...")
        
        results = []
        for i, sample in enumerate(tqdm(test_samples, desc="Testing ToolBench model")):
            query = sample['query']
            relevant_apis = sample.get('relevant_apis', [])
            
            # Get query embedding using ToolBench model
            query_embedding = self.get_toolbench_embedding(query)
            
            # Search for similar APIs using ToolBench vector database
            similar_apis = self.search_similar_apis_toolbench(query_embedding, similarity_threshold, top_k)
            
            # Evaluate predictions
            metrics = self.evaluate_predictions(similar_apis, relevant_apis)
            
            results.append({
                'query': query,
                'predicted_apis': similar_apis,
                'relevant_apis': relevant_apis,
                'metrics': metrics
            })
        
        # Calculate aggregate metrics
        aggregate_metrics = self.calculate_aggregate_metrics(results)
        
        return {
            'results': results,
            'aggregate_metrics': aggregate_metrics
        }
    
    def test_trained_model(self, test_samples: List[Dict[str, Any]], similarity_threshold: float = 0.5, top_k: int = 100) -> Dict[str, Any]:
        """Test trained model performance"""
        print(f"\n🧪 Testing trained retriever model (threshold={similarity_threshold}, top_k={top_k})...")
        
        results = []
        for i, sample in enumerate(tqdm(test_samples, desc="Testing trained model")):
            query = sample['query']
            relevant_apis = sample.get('relevant_apis', [])
            
            # Get query embedding using trained model
            query_embedding = self.get_trained_embedding(query)
            
            # Search for similar APIs using trained model vector database
            similar_apis = self.search_similar_apis_trained(query_embedding, similarity_threshold, top_k)
            
            # Evaluate predictions
            metrics = self.evaluate_predictions(similar_apis, relevant_apis)
            
            results.append({
                'query': query,
                'predicted_apis': similar_apis,
                'relevant_apis': relevant_apis,
                'metrics': metrics
            })
        
        # Calculate aggregate metrics
        aggregate_metrics = self.calculate_aggregate_metrics(results)
        
        return {
            'results': results,
            'aggregate_metrics': aggregate_metrics
        }
    
    def calculate_aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate metrics from individual results"""
        metrics_list = [r['metrics'] for r in results]
        
        return {
            'precision': np.mean([m['precision'] for m in metrics_list]),
            'recall': np.mean([m['recall'] for m in metrics_list]),
            'f1_score': np.mean([m['f1_score'] for m in metrics_list]),
            'avg_predicted_count': np.mean([m['predicted_count'] for m in metrics_list]),
            'avg_relevant_count': np.mean([m['relevant_count'] for m in metrics_list]),
            'avg_intersection_count': np.mean([m['intersection_count'] for m in metrics_list])
        }
    
    def print_comparison_results(self, toolbench_results: Dict[str, Any], trained_results: Dict[str, Any]):
        """Print comparison results"""
        print("\n" + "="*80)
        print("📊 MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        toolbench_metrics = toolbench_results['aggregate_metrics']
        trained_metrics = trained_results['aggregate_metrics']
        
        print(f"{'Metric':<20} {'ToolBench':<15} {'Trained':<15} {'Difference':<15}")
        print("-" * 80)
        print(f"{'Precision':<20} {toolbench_metrics['precision']:<15.4f} {trained_metrics['precision']:<15.4f} {trained_metrics['precision'] - toolbench_metrics['precision']:<15.4f}")
        print(f"{'Recall':<20} {toolbench_metrics['recall']:<15.4f} {trained_metrics['recall']:<15.4f} {trained_metrics['recall'] - toolbench_metrics['recall']:<15.4f}")
        print(f"{'F1 Score':<20} {toolbench_metrics['f1_score']:<15.4f} {trained_metrics['f1_score']:<15.4f} {trained_metrics['f1_score'] - toolbench_metrics['f1_score']:<15.4f}")
        print(f"{'Avg Predicted':<20} {toolbench_metrics['avg_predicted_count']:<15.2f} {trained_metrics['avg_predicted_count']:<15.2f} {trained_metrics['avg_predicted_count'] - toolbench_metrics['avg_predicted_count']:<15.2f}")
        print(f"{'Avg Relevant':<20} {toolbench_metrics['avg_relevant_count']:<15.2f} {trained_metrics['avg_relevant_count']:<15.2f} {'N/A':<15}")
        print(f"{'Avg Intersection':<20} {toolbench_metrics['avg_intersection_count']:<15.2f} {trained_metrics['avg_intersection_count']:<15.2f} {trained_metrics['avg_intersection_count'] - toolbench_metrics['avg_intersection_count']:<15.2f}")
        
        print("\n" + "="*80)
        print("🏆 WINNER ANALYSIS")
        print("="*80)
        
        if trained_metrics['f1_score'] > toolbench_metrics['f1_score']:
            print("🎉 Trained model wins in F1 Score!")
        elif trained_metrics['f1_score'] < toolbench_metrics['f1_score']:
            print("🎉 ToolBench model wins in F1 Score!")
        else:
            print("🤝 Models are tied in F1 Score!")
        
        if trained_metrics['precision'] > toolbench_metrics['precision']:
            print("🎯 Trained model has higher precision!")
        elif trained_metrics['precision'] < toolbench_metrics['precision']:
            print("🎯 ToolBench model has higher precision!")
        
        if trained_metrics['recall'] > toolbench_metrics['recall']:
            print("📈 Trained model has higher recall!")
        elif trained_metrics['recall'] < toolbench_metrics['recall']:
            print("📈 ToolBench model has higher recall!")
    
    def load_test_instruction_data(self, instruction_dir: str = TEST_INSTRUCTION_DIR) -> List[Dict[str, Any]]:
        """
        Load test instruction data from *_instruction.json files
        """
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
        
        for file in tqdm(instruction_files, desc="Loading instruction files"):
            file_path = os.path.join(instruction_dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract samples with queries and relevant APIs
                for sample in data:
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
    
    def run_comparison_with_different_thresholds(self, max_samples: int = None):
        """Run performance comparison with different similarity thresholds"""
        print("🚀 Starting Model Performance Comparison")
        print("=" * 50)
        
        # Load test data
        test_samples = self.load_test_instruction_data()
        
        if not test_samples:
            print("❌ No test samples loaded. Exiting.")
            return
        
        if max_samples:
            test_samples = test_samples[:max_samples]
        
        print(f"📊 Testing on {len(test_samples)} samples")
        
        # Test with different similarity thresholds and top_k values
        thresholds = [0.3, 0.5, 0.7]
        top_k_values = [5, 10, 20]
        
        all_results = {}
        
        for threshold in thresholds:
            for top_k in top_k_values:
                print(f"\n{'='*60}")
                print(f"Testing with threshold={threshold}, top_k={top_k}")
                print(f"{'='*60}")
                
                # Test ToolBench model
                toolbench_results = self.test_toolbench_model(test_samples, threshold, top_k)
                
                # Test trained model
                trained_results = self.test_trained_model(test_samples, threshold, top_k)
                
                # Print comparison
                self.print_comparison_results(toolbench_results, trained_results)
                
                # Store results
                key = f"threshold_{threshold}_topk_{top_k}"
                all_results[key] = {
                    'toolbench_results': toolbench_results,
                    'trained_results': trained_results
                }
        
        # Save detailed results
        output_path = os.path.join(BASE_DIR, "new_retriever_comparison_results.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Detailed results saved to: {output_path}")
        
        return all_results
    
    def run_single_comparison(self, similarity_threshold: float = 0.5, top_k: int = 10, max_samples: int = None):
        """Run single performance comparison with specified parameters"""
        print("🚀 Starting Model Performance Comparison")
        print("=" * 50)
        
        # Load test data
        test_samples = self.load_test_instruction_data()
        
        if not test_samples:
            print("❌ No test samples loaded. Exiting.")
            return
        
        if max_samples:
            test_samples = test_samples[:max_samples]
        
        print(f"📊 Testing on {len(test_samples)} samples")
        print(f"📊 Parameters: threshold={similarity_threshold}, top_k={top_k}")
        
        # Test ToolBench model
        toolbench_results = self.test_toolbench_model(test_samples, similarity_threshold, top_k)
        
        # Test trained model
        trained_results = self.test_trained_model(test_samples, similarity_threshold, top_k)
        
        # Print comparison
        self.print_comparison_results(toolbench_results, trained_results)
        
        # Save detailed results
        output_path = os.path.join(BASE_DIR, "new_retriever_comparison_results_single.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'toolbench_results': toolbench_results,
                'trained_results': trained_results,
                'parameters': {
                    'similarity_threshold': similarity_threshold,
                    'top_k': top_k,
                    'num_samples': len(test_samples)
                }
            }, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Detailed results saved to: {output_path}")
        
        return {
            'toolbench_results': toolbench_results,
            'trained_results': trained_results
        }


def main():
    """Main comparison function"""
    print("🧪 New Retriever Model Performance Comparison")
    print("=" * 50)
    
    # Initialize comparator
    comparator = NewRetrieverPerformanceComparator()
    
    # Load models and vector database
    if not comparator.load_models():
        print("❌ Failed to load models. Exiting.")
        return
    
    if not comparator.load_vector_database():
        print("❌ Failed to load vector database. Exiting.")
        return
    
    # Parse command line arguments
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "multi":
            # Run comparison with multiple thresholds
            print(f"\n📊 Running multi-threshold comparison...")
            comparator.run_comparison_with_different_thresholds(max_samples=100)
        else:
            # Run single comparison with custom parameters
            threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
            top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 100
            
            print(f"\n📊 Running single comparison with threshold={threshold}, top_k={top_k}...")
            comparator.run_single_comparison(threshold, top_k, max_samples)
    else:
        # Default single comparison
        print(f"\n📊 Running default single comparison...")
        comparator.run_single_comparison(max_samples=100)
    
    print("\n✅ Comparison completed!")


if __name__ == "__main__":
    main() 