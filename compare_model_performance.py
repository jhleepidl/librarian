#!/usr/bin/env python3
"""
Compare performance between ToolBench model and trained query embedding model
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
from train_query_embedding import QueryEmbeddingModel
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_MODEL_PATH = "/home/jhlee/librarian/trained_query_model/best_model.pt"
TOOLBENCH_MODEL_NAME = "ToolBench/ToolBench_IR_bert_based_uncased"
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db_toolbench")
TEST_INSTRUCTION_DIR = "/home/jhlee/ToolBench/data/test_instruction"


class ModelPerformanceComparator:
    """
    Compare performance between ToolBench model and trained query embedding model
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.toolbench_model = None
        self.trained_model = None
        self.vector_db = None
        self.api_metadata = []
        
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
        
        # Load trained query embedding model
        try:
            self.trained_model = QueryEmbeddingModel()
            if os.path.exists(BEST_MODEL_PATH):
                checkpoint = torch.load(BEST_MODEL_PATH, map_location=self.device)
                self.trained_model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Trained query model loaded from {BEST_MODEL_PATH}")
            else:
                print(f"⚠️  No trained model found at {BEST_MODEL_PATH}")
                return False
            
            self.trained_model = self.trained_model.to(self.device)
            self.trained_model.eval()
        except Exception as e:
            print(f"❌ Error loading trained model: {e}")
            return False
        
        return True
    
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
    
    def get_toolbench_embedding(self, query: str) -> np.ndarray:
        """Get query embedding using ToolBench model"""
        with torch.no_grad():
            embedding = self.toolbench_model.encode([query], convert_to_tensor=True, device=self.device)
            # Normalize for cosine similarity
            normalized_embedding = F.normalize(embedding, p=2, dim=-1)
            return normalized_embedding.cpu().numpy()
    
    def get_trained_embedding(self, query: str) -> np.ndarray:
        """Get query embedding using trained model"""
        tokenizer = self.trained_model.base_model.tokenizer
        max_length = self.trained_model.base_model.get_max_seq_length()
        
        features = tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        features = {k: v.to(self.device) for k, v in features.items()}
        
        with torch.no_grad():
            embedding = self.trained_model(features["input_ids"], features["attention_mask"])
            # Normalize for cosine similarity
            normalized_embedding = F.normalize(embedding, p=2, dim=-1)
            return normalized_embedding.cpu().numpy()
    
    def search_similar_apis(self, query_embedding: np.ndarray, similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Search for similar APIs in vector database"""
        # Search in vector database
        scores, indices = self.vector_db.search(query_embedding, k=100)  # Get top 100
        
        # Filter by similarity threshold
        similar_apis = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= similarity_threshold:
                api_info = self.api_metadata[idx].copy()
                api_info['similarity_score'] = float(score)
                similar_apis.append(api_info)
        
        return similar_apis
    
    def iterative_residual_search(self, query: str, max_iterations: int = 10, residual_threshold: float = 0.5, beam_size: int = 3) -> List[Dict[str, Any]]:
        """
        Iterative residual-based search using trained model with beam search
        """
        # Initialize beam with original query embedding
        initial_embedding = self.get_trained_embedding(query)
        
        # Beam state: (apis, current_embedding, total_similarity_score)
        beam = [([], initial_embedding, 0.0)]
        
        for iteration in range(max_iterations):
            new_beam = []
            
            for apis, current_embedding, total_score in beam:
                # Search for similar APIs
                similar_apis = self.search_similar_apis(current_embedding, similarity_threshold=0.1)
                
                if not similar_apis:
                    new_beam.append((apis, current_embedding, total_score))
                    continue
                
                # Get top beam_size candidates
                top_candidates = similar_apis[:beam_size]
                
                for candidate_api in top_candidates:
                    # Get the API's embedding
                    api_text = f"{candidate_api['tool_name']} {candidate_api['api_name']} {candidate_api['api_description']}"
                    api_embedding = self.toolbench_model.encode([api_text], convert_to_tensor=True, device=self.device)
                    api_embedding = F.normalize(api_embedding, p=2, dim=-1).cpu().numpy()
                    
                    # Calculate residual
                    residual = current_embedding - api_embedding
                    residual_norm = np.linalg.norm(residual)
                    
                    # Calculate new total score (higher similarity = better)
                    new_total_score = total_score + candidate_api['similarity_score']
                    
                    # Add to new beam
                    new_apis = apis + [candidate_api]
                    new_beam.append((new_apis, residual, new_total_score))
                    
                    print(f"  Iteration {iteration + 1}: API={candidate_api['tool_name']}/{candidate_api['api_name']}, "
                          f"Similarity={candidate_api['similarity_score']:.4f}, Residual_norm={residual_norm:.4f}, "
                          f"Total_score={new_total_score:.4f}")
            
            # Keep top beam_size candidates based on total score
            new_beam.sort(key=lambda x: x[2], reverse=True)
            beam = new_beam[:beam_size]
            
            # Check if all residuals are small enough
            all_small_residuals = all(np.linalg.norm(state[1]) < residual_threshold for state in beam)
            if all_small_residuals:
                print(f"  All residual norms below threshold ({residual_threshold}), stopping")
                break
        
        # Return the best candidate (highest total score)
        if beam:
            best_apis, _, best_score = max(beam, key=lambda x: x[2])
            print(f"  Best beam candidate with total score: {best_score:.4f}")
            return best_apis
        else:
            return []
    
    def iterative_residual_search_greedy(self, query: str, max_iterations: int = 10, residual_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Original greedy iterative residual-based search using trained model
        """
        predicted_apis = []
        current_embedding = self.get_trained_embedding(query)
        
        for iteration in range(max_iterations):
            # Search for most similar API
            similar_apis = self.search_similar_apis(current_embedding, similarity_threshold=0.1)
            
            if not similar_apis:
                break
            
            # Get the most similar API
            best_api = similar_apis[0]
            predicted_apis.append(best_api)
            
            # Get the API's embedding
            api_text = f"{best_api['tool_name']} {best_api['api_name']} {best_api['api_description']}"
            api_embedding = self.toolbench_model.encode([api_text], convert_to_tensor=True, device=self.device)
            api_embedding = F.normalize(api_embedding, p=2, dim=-1).cpu().numpy()
            
            # Calculate residual
            residual = current_embedding - api_embedding
            residual_norm = np.linalg.norm(residual)
            
            print(f"  Iteration {iteration + 1}: API={best_api['tool_name']}/{best_api['api_name']}, "
                  f"Similarity={best_api['similarity_score']:.4f}, Residual_norm={residual_norm:.4f}")
            
            # Check if residual is small enough
            if residual_norm < residual_threshold:
                print(f"  Residual norm ({residual_norm:.4f}) below threshold ({residual_threshold}), stopping")
                break
            
            # Update current embedding to residual
            current_embedding = residual
        
        return predicted_apis
    
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
    
    def test_toolbench_model(self, test_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test ToolBench model performance"""
        print("\n🧪 Testing ToolBench model...")
        
        results = []
        for i, sample in enumerate(tqdm(test_samples, desc="Testing ToolBench model")):
            query = sample['query']
            relevant_apis = sample.get('relevant_apis', [])
            
            # Get query embedding using ToolBench model
            query_embedding = self.get_toolbench_embedding(query)
            
            # Search for similar APIs
            similar_apis = self.search_similar_apis(query_embedding, similarity_threshold=0.5)
            
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
    
    def test_trained_model(self, test_samples: List[Dict[str, Any]], use_beam_search: bool = True) -> Dict[str, Any]:
        """Test trained model performance using iterative residual search"""
        search_method = "beam search" if use_beam_search else "greedy search"
        print(f"\n🧪 Testing trained model (iterative residual {search_method})...")
        
        results = []
        for i, sample in enumerate(tqdm(test_samples, desc=f"Testing trained model ({search_method})")):
            query = sample['query']
            relevant_apis = sample.get('relevant_apis', [])
            
            # Use iterative residual search (beam or greedy)
            if use_beam_search:
                predicted_apis = self.iterative_residual_search(query, beam_size=3)
            else:
                predicted_apis = self.iterative_residual_search_greedy(query)
            
            # Evaluate predictions
            metrics = self.evaluate_predictions(predicted_apis, relevant_apis)
            
            results.append({
                'query': query,
                'predicted_apis': predicted_apis,
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
    
    def run_comparison(self, max_samples: int = None):
        """Run complete performance comparison"""
        print("🚀 Starting Model Performance Comparison")
        print("=" * 50)
        
        # Load test data from ToolBench instruction files
        test_samples = self.load_test_instruction_data()
        
        if max_samples:
            test_samples = test_samples[:max_samples]
        
        print(f"📊 Testing on {len(test_samples)} samples from ToolBench test instructions")
        
        # Test ToolBench model
        toolbench_results = self.test_toolbench_model(test_samples)
        
        # Test trained model with beam search
        trained_beam_results = self.test_trained_model(test_samples, use_beam_search=True)
        
        # Test trained model with greedy search
        trained_greedy_results = self.test_trained_model(test_samples, use_beam_search=False)
        
        # Print comparison
        print("\n" + "="*80)
        print("📊 COMPARISON: ToolBench vs Trained Model (Beam Search)")
        print("="*80)
        self.print_comparison_results(toolbench_results, trained_beam_results)
        
        print("\n" + "="*80)
        print("📊 COMPARISON: ToolBench vs Trained Model (Greedy Search)")
        print("="*80)
        self.print_comparison_results(toolbench_results, trained_greedy_results)
        
        print("\n" + "="*80)
        print("📊 COMPARISON: Beam Search vs Greedy Search")
        print("="*80)
        self.print_comparison_results(trained_greedy_results, trained_beam_results)
        
        # Save detailed results
        output_path = "/home/jhlee/librarian/model_comparison_results_toolbench.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'toolbench_results': toolbench_results,
                'trained_beam_results': trained_beam_results,
                'trained_greedy_results': trained_greedy_results
            }, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Detailed results saved to: {output_path}")


def main():
    """Main comparison function"""
    print("🧪 Model Performance Comparison")
    print("=" * 50)
    
    # Initialize comparator
    comparator = ModelPerformanceComparator()
    
    # Load models and vector database
    if not comparator.load_models():
        print("❌ Failed to load models. Exiting.")
        return
    
    if not comparator.load_vector_database():
        print("❌ Failed to load vector database. Exiting.")
        return
    
    # Run comparison on ToolBench test instruction data
    print(f"\n📊 Running comparison on ToolBench test instruction data...")
    comparator.run_comparison(max_samples=100)
    
    print("\n✅ Comparison completed!")


if __name__ == "__main__":
    main() 